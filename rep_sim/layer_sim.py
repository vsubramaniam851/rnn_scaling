import torch
import torchvision
import torch.distributed as dist
import time

from .rep_sims import CKA, Procrustes, DifferentiableRSA, Ridge
from .layer_extract import FeatureMapExtractor

def torchvision_fe(model, inputs, device):
    layers = torchvision.models.feature_extraction.get_graph_node_names(model)[0]
    extract_layers = [l for l in layers if ('mlp' in l) or ('self_attention' in l)] + ['getitem_5']
    feature_extractor = torchvision.models.feature_extraction.create_feature_extractor(model, return_nodes = extract_layers)
    feature_extractor = feature_extractor.to(device)
    with torch.no_grad():
        output = feature_extractor(inputs)
    return output

def layer_supervision(target_model_layers, student_model_layers):
    source_count = len(target_model_layers)
    target_count = len(student_model_layers)
    step = (target_count - 1) / (source_count - 1) if source_count > 1 else 1

    mapping = {}
    for i, source_layer in enumerate(target_model_layers):
        target_index = min(round(i * step), target_count - 1)
        target_layer = student_model_layers[target_index]
        mapping[source_layer] = target_layer
    return mapping

def get_layer_outputs(model, inputs, device, eval = True, **kwargs):
    extractor = FeatureMapExtractor(model, eval = eval, device = device, enforce_input_shape = True)
    if kwargs['lengths'] != None:
        feature_maps, outputs = extractor.get_feature_maps(inputs, **kwargs)
    else:
        feature_maps, outputs = extractor.get_feature_maps(inputs)
    return feature_maps, outputs

def layermap_sim(train_model, target_model, student_model, rep_sim, inputs, target_inputs, device, lengths = None, torchvision_extract = False, 
                 token_sim = False, is_main_process = False):
    cka = CKA(device)
    diff_rsa = DifferentiableRSA(device)
    ridge = Ridge(device)
    if not torchvision_extract:
        pretrained_outputs, _ = get_layer_outputs(target_model, target_inputs, device, eval = True, lengths = lengths)
    else:
        pretrained_outputs = torchvision_fe(target_model, inputs, device)
    training_outputs, outputs = get_layer_outputs(train_model, inputs, device, eval = False, lengths = lengths)
    
    is_distributed = False
    rank = 0
    world_size = 1
    try:
        is_distributed = dist.is_initialized()
        if is_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except:
        pass

    teacher_layers = list(pretrained_outputs.keys())
    student_layers = list(training_outputs.keys())
    if len(teacher_layers) <= len(student_layers):
        model_mapping = layer_supervision(teacher_layers, student_layers)
    else:
        model_mapping = layer_supervision(teacher_layers, student_layers)
    sim_scores = {}
    for layer in model_mapping:
        assert layer in pretrained_outputs, f'Layer {layer} is not in target network {pretrained_outputs.keys()}'
        tr_layer = model_mapping[layer]
        assert tr_layer in training_outputs, f'Layer {layer} is not in {student_model} {training_outputs.keys()}'

        pretrained_output = pretrained_outputs[layer]
        if isinstance(pretrained_output, tuple):
            pretrained_output = pretrained_output[0]
        training_output = training_outputs[tr_layer]
        assert pretrained_output.shape[0] == training_output.shape[0], f'Guide network shape: {pretrained_output.shape}, Target network shape: {training_output.shape}'

        if rep_sim == 'CKA':
            if is_distributed:
                rank = dist.get_rank()
                pretrained_output = pretrained_output.flatten(start_dim = 1)
                training_output = training_output.flatten(start_dim = 1)
                
                gathered_pretrained = [torch.zeros_like(pretrained_output) for _ in range(world_size)]
                gathered_training = [torch.zeros_like(training_output) for _ in range(world_size)]

                with torch.no_grad():
                    dist.all_gather(gathered_pretrained, pretrained_output)
                    dist.all_gather(gathered_training, training_output)
                gathered_pretrained[rank] = pretrained_output
                gathered_training[rank] = training_output
                pretrained_output = torch.cat(gathered_pretrained, dim = 0)
                training_output = torch.cat(gathered_training, dim = 0)
                sim = 1 - cka.linear_CKA(training_output.to(torch.float32), pretrained_output.to(torch.float32))
            else:
                pretrained_output = pretrained_output.flatten(start_dim = 1)
                training_output = training_output.flatten(start_dim = 1)
                sim = 1 - cka.linear_CKA(training_output.to(torch.float32), pretrained_output.to(torch.float32))
        elif rep_sim == 'RSA':
            pretrained_output = pretrained_output.flatten(start_dim=1)
            training_output = training_output.flatten(start_dim=1)
            sim = 1 - diff_rsa.rsa(training_output.to(torch.float32), pretrained_output.to(torch.float32))
        elif rep_sim == 'Ridge':
            pretrained_output = pretrained_output.flatten(start_dim=1)
            training_output = training_output.contiguous().flatten(start_dim=1)
            sim = 1 - ridge.linear_regression_similarity(training_output.to(torch.float32), pretrained_output.to(torch.float32))
        else:
            raise NotImplementedError()
        sim_scores[tr_layer] = sim
    del pretrained_outputs
    del training_outputs
    return sim_scores, outputs