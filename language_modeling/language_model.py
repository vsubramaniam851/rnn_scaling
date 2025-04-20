import os
import numpy as np
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from .args import *
from .rnn_models import RNNLM
from .transformer import TransformerLM
from .webtext import make_webtext_dataloaders
from rep_sim import rep_similarity_loss

def init_distributed():
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend = 'nccl')
    torch.cuda.set_device(local_rank)
    
    return local_rank

def total_loss(train_model, target_model, rep_sim, loss_fn, inputs, rep_sim_alpha, device, student_model = 'LSTM', 
               lengths = None, use_noise = False, is_main_process = False):
    sim_loss, sim_dict, outputs = rep_similarity_loss(train_model, target_model, rep_sim, inputs, device, student_model = student_model, 
                                                    lengths = lengths, use_noise = use_noise, is_main_process = is_main_process)

    lm_logits = outputs[0]
    shift_logits, shift_labels, _ = shift_logits_labels(inputs, lm_logits, None)
    ce_sum = loss_fn(shift_logits, shift_labels)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ce_sum, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    local_tokens = shift_labels.numel()
    global_tokens = local_tokens * world_size
    ce_loss = ce_sum / global_tokens
    return ce_loss + rep_sim_alpha * sim_loss, sim_loss, ce_loss, outputs[-1]

def avg_step_size(model, before_state_dict):
    sum_changes = 0
    count = 0
    with torch.no_grad():
        if isinstance(model, DDP):
            after_state_dict = model.module.state_dict()
        else:
            after_state_dict = model.state_dict()
        for key in before_state_dict:
            change = (after_state_dict[key] - before_state_dict[key]).abs().mean().item()
            sum_changes += change
            count += 1
    return sum_changes / count

def shift_logits_labels(input_ids, lm_logits, lengths):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = input_ids[..., 1:].contiguous()
    shift_labels = shift_labels.view(-1)
    return shift_logits, shift_labels, lengths

def load_student_model(student_model, vocab_size, embedding_dim, hidden_dim, num_layers, d_model, trans_layers, nhead, d_fd, context_length, device):
    if student_model == 'LSTM':
        model = RNNLM(student_model, vocab_size = vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
                      num_layers = num_layers, device = device)
    elif student_model == 'RNN':
        model = RNNLM(student_model, vocab_size = vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
                      num_layers = num_layers, device = device)
    elif student_model == 'Transformer':
        model = TransformerLM(vocab_size, d_model = d_model, nhead = nhead, num_layers = trans_layers, dim_feedforward = d_fd, seq_len = context_length)
    else:
        raise NotImplementedError()
    model = model.to(device)
    return model

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def validate(model, valid_loader, loss_fn, device):
    model.eval()
    valid_loss = 0.0
    hidden = None
    for batch in tqdm(valid_loader, desc = 'Iterating over validation set...'):
        input_ids = batch['input_ids'].to(device)
        lengths = batch.get('lengths', None)
        with torch.no_grad():
            lm_logits, hidden = model(input_ids, hidden = None, lengths = lengths)
        shift_logits, shift_labels, lengths = shift_logits_labels(input_ids, lm_logits, batch, lengths, device)
        loss = loss_fn(shift_logits, shift_labels)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss, op = dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        local_tokens = shift_labels.numel()
        global_tokens = local_tokens * world_size
        loss = loss / global_tokens
        valid_loss += loss.item()
    avg_val_loss = valid_loss / len(valid_loader)
    return avg_val_loss

def train_lm(args, exp_name, repr_sim, student_model, context_length = 256,
             batch_size = 64, lr = 1e-3, accumulation = 1, embedding_dim = 256, hidden_dim = 512, num_layers = 4,
             d_model = 512, nhead = 16, trans_layers = 4, d_fd = 2048, rep_dist = None, rep_sim_alpha = 1.0):
    
    local_rank = init_distributed() if args.distributed else 0
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main_process = local_rank == 0

    if is_main_process:
        wandb.init(
            project = exp_name,
            config = {
                'model': student_model,
                'repr-sim': repr_sim,
                'lr': lr,
                'batch_size': batch_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'fc_dim': fc_dim,
                'rep_dist': rep_dist
            }
        )

    train_loader, valid_loader, vocab_size = make_webtext_dataloaders(batch_size, seq_len = context_length, reload_dataset = args.reload_dataset, 
                                                                      distributed=args.distributed)
    loss_fn = nn.CrossEntropyLoss(ignore_index = -1, reduction = 'sum')
    
    model = load_student_model(student_model, vocab_size, embedding_dim, hidden_dim, num_layers, d_model, trans_layers, nhead, d_fd, context_length,
                               device)
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f'Compilation error {e}')

    if repr_sim:
        target_model = TransformerLM(vocab_size = vocab_size, d_model = d_model, nhead = nhead, num_layers = trans_layers, seq_len = 256, dim_feedforward = d_fd).to(device)
        try:
            target_model = torch.compile(target_model)
        except Exception as e:
            print(f'Compilation error {e}')

    if args.distributed:
        model = DDP(model, device_ids = [local_rank], output_device = local_rank, find_unused_parameters = False)
        if repr_sim:
            target_model = DDP(target_model, device_ids = [local_rank], output_device = local_rank, find_unused_parameters = False)

    optimizer = optim.AdamW(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader), eta_min = lr/10)
    val_losses = []
    step_train_losses = []
    step_sizes = []
    step_ce_loss = []
    step_rep_sim_loss = []

    for i, batch in enumerate(tqdm(train_loader, desc = 'Iterating over pretraining set...')):
        if i % 50000 == 0:
            avg_val_loss = validate(model, valid_loader, loss_fn, device)
            val_losses.append(avg_val_loss)
            if is_main_process:
                wandb.log({'val_loss': avg_val_loss})
            if avg_val_loss <= min(val_losses):
                if isinstance(model, DDP):
                    torch.save(model.module.state_dict(), f'saved_models/{exp_name}.pt')
                else:
                    torch.save(model.state_dict(), f'saved_models/{exp_name}.pt')
                    print(f'Validation Loss {avg_val_loss}')

        if i * batch_size * context_length > args.token_budget and args.token_budget != -1:
            break

        model.train()
        train_loss = 0.0
        hidden = None
        input_ids = batch['input_ids'].to(device, non_blocking = True)
        lengths = batch.get('lengths', None)
        if not repr_sim:
            lm_logits, hidden = model(input_ids, hidden = None, lengths = lengths)
            shift_logits, shift_labels, lengths = shift_logits_labels(input_ids, lm_logits, lengths)
            loss = loss_fn(shift_logits, shift_labels)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss, op = dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
            else:
                world_size = 1
            local_tokens = loss.numel()
            global_tokens = local_tokens * world_size
            loss = loss / global_tokens
            ce_loss = None
        else:
            loss, sim_loss, ce_loss, hidden = total_loss(train_model = model, target_model = target_model, rep_sim = rep_dist, 
                                                        loss_fn = loss_fn, inputs =  input_ids, rep_sim_alpha = rep_sim_alpha, 
                                                        device = device, student_model = student_model, lengths = lengths, is_main_process = is_main_process)

            step_ce_loss.append(ce_loss.item())
            step_rep_sim_loss.append(sim_loss.item())
            if i % 20 == 0:
                if is_main_process:
                    avg_ce_loss = np.mean(step_ce_loss[-20:])
                    avg_rep_sim_loss = np.mean(step_rep_sim_loss[-20:])
                    wandb.log({'ce_loss': avg_ce_loss, 'rep_sim_loss': avg_rep_sim_loss})
        if isinstance(model, DDP):
            before_update_params = {name: param.clone() for name, param in model.module.named_parameters()}
        else:
            before_update_params = {name: param.clone() for name, param in model.named_parameters()}
            
        loss.backward()
        if i % 10000 == 0:   
            grad_norm = get_grad_norm(model)
            if is_main_process:
                wandb.log({'grad_norm': grad_norm})
        if isinstance(model, DDP):
            nn.utils.clip_grad_norm_(model.module.parameters(), max_norm = 0.25)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.25)
        if (i + 1) % accumulation == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if ce_loss == None:
            train_loss += loss.item()
        else:
            train_loss += ce_loss.item()
        step_train_losses.append(loss.item())
        hidden = tuple(h.detach() for h in hidden)
        step_size = avg_step_size(model, before_update_params)
        step_sizes.append(step_size)

        if i % 20 == 0 and is_main_process:
            avg_train_loss = np.mean(step_train_losses[-20:])
            wandb.log({'train_loss': avg_train_loss, 'step_size': step_size})
            wandb.log({'lr': optimizer.param_groups[0]['lr']})
        if i % 5000 == 0 and is_main_process:
            avg_train_loss = np.mean(step_train_losses[-1000:])
            print(f'Step {i + 1}, Training Loss {avg_train_loss}')

    avg_val_loss = validate(model, valid_loader, loss_fn, device)
    val_losses.append(avg_val_loss)
    if is_main_process:
        wandb.log({'val_loss': avg_val_loss})
    if avg_val_loss <= min(val_losses):
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), f'saved_models/{exp_name}.pt')
        else:
            torch.save(model.state_dict(), f'saved_models/{exp_name}.pt')
            print(f'Validation Loss {avg_val_loss}')

    if is_main_process:
        if not os.path.exists(f'{args.logging}/{args.exp_name}'):
            os.makedirs(f'{args.logging}/{args.exp_name}')
        with open(f'{args.logging}/{exp_name}/args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        loss_info = {'step_train_losses': step_train_losses, 'step_sizes': step_sizes, 'val_losses': val_losses, 
                    'step_ce_loss': step_ce_loss, 'step_rep_sim_loss': step_rep_sim_loss}
        loss_info = {key: value for key, value in loss_info.items() if value != []}
        with open(f'{args.logging}/{exp_name}/info.json', 'w') as f:
            json.dump(loss_info, f)

        wandb.finish()
    if args.distributed:
        dist.barrier()
    if isinstance(model, DDP):
        model = model.module

    return model, step_train_losses, val_losses, step_ce_loss, step_rep_sim_loss

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, step_train_losses, val_losses, step_ce_loss, step_rep_sim_loss = train_lm(args, exp_name = args.exp_name, repr_sim = args.rep_sim, 
                                                                                     student_model = args.student_model, context_length = args.context_length, 
                                                                                     batch_size = args.batch_size, lr = args.lr, accumulation = args.accumulation, 
                                                                                     embedding_dim = args.embedding_dim, hidden_dim = args.hidden_dim, num_layers = args.num_layers, 
                                                                                     d_model = args.d_model, nhead = args.nheads, trans_layers = args.trans_layers, 
                                                                                     d_fd = args.d_fd, rep_dist = args.rep_dist, rep_sim_alpha = args.rep_sim_alpha)