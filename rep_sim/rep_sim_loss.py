import os
import torch

from .layer_sim import *

def create_rand_inputs(inputs):
    if inputs.dtype == torch.float32:
        mean = 0.0
        std = 0.1
        new_inputs = torch.randn(*inputs.shape)
        noise = torch.randn(*inputs.shape) * std + mean
        noisy_batch = new_inputs + noise
        noisy_batch = torch.clamp(noisy_batch, 0.0, 1.0)
    elif inputs.dtype == torch.long:
        max_val = torch.max(inputs).item()
        noisy_batch = torch.randint(0, max_val + 1, inputs.shape)
    else:
        raise ValueError(f'Data type {inputs.dtype} is not compatible...')
    return noisy_batch

def rep_similarity_loss(train_model, target_model, rep_sim, inputs, device, student_model = 'ResNet-50', lengths = None, use_noise = False, 
                        torchvision_extract = False, token_sim = False, is_main_process = False):
    if use_noise:
        target_batch = create_rand_inputs(inputs)
        target_batch = target_batch.to(device)
    else:
        target_batch = inputs

    sim_dict = {}
    if rep_sim == 'CKA':
        sim_dict, logits = layermap_sim(train_model, target_model, student_model, rep_sim, inputs, target_batch, device, lengths = lengths,
                torchvision_extract = torchvision_extract, token_sim = token_sim, is_main_process = is_main_process)
        sims = torch.stack(list(sim_dict.values()))
        sim = torch.sum(sims)
    elif rep_sim == 'RSA':
        sim_dict, logits = torch.stack(list(layermap_sim(train_model, target_model, student_model, rep_sim, inputs, target_batch, device, lengths = lengths).values()))
        sim = torch.sum(sim_dict)
    elif rep_sim == 'Procrustes':
        sim_dict, logits = torch.stack(list(layermap_sim(train_model, target_model, rep_sim, inputs, target_batch, device, lengths = lengths).values()))
        sim = torch.sum(sim_dict)
    elif rep_sim == 'Ridge':
        sim_dict, logits = torch.stack(list(layermap_sim(train_model, target_model, student_model, rep_sim, inputs, target_batch, device, lengths = lengths).values()))
        sim = torch.sum(sim_dict)
    else:
        raise NotImplementedError
    return sim, sim_dict, logits
