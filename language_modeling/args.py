import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', action = 'store', type = str, required = True, dest = 'exp_name')
    parser.add_argument('--seed', action = 'store', type = int, dest = 'seed', default = 0)
    
    parser.add_argument('--eval', action = 'store_true', dest = 'eval')
    parser.add_argument('--student_model', action = 'store', type = str, dest = 'student_model', default = 'RNN', choices = ['LSTM', 'RNN', 'Transformer'])
    parser.add_argument('--context_length', action = 'store', type = int, dest = 'context_length', default = 128)
    parser.add_argument('--reload_dataset', action = 'store_true', dest = 'reload_dataset')

    parser.add_argument('--num_workers', action = 'store', type = int, dest = 'num_workers', default = 24)
    parser.add_argument('--batch_size', action = 'store', type = int, dest = 'batch_size', default = 256)

    parser.add_argument('--lr', action = 'store', type = float, dest = 'lr', default = 1e-4)
    parser.add_argument('--grad_acc', action = 'store', type = int, dest = 'accumulation', default = 1)

    parser.add_argument('--embedding_dim', action = 'store', type = int, dest = 'embedding_dim', default = 1300)
    parser.add_argument('--hidden_dim', action = 'store', type = int, dest = 'hidden_dim', default = 1300)
    parser.add_argument('--num_layers', action = 'store', type = int, dest = 'num_layers', default = 12)

    parser.add_argument('--d_model', action = 'store', type = int, dest = 'd_model', default = 768)
    parser.add_argument('--trans_layers', action = 'store', type = int, dest = 'trans_layers', default = 6)
    parser.add_argument('--nheads', action = 'store', type = int, dest = 'nheads', default = 16)
    parser.add_argument('--d_fd', action = 'store', type = int, dest = 'd_fd', default = 3072)

    parser.add_argument('--rep_sim', action = 'store_true', dest = 'rep_sim')
    parser.add_argument('--repdist', action = 'store', type = str, dest = 'rep_dist', default = 'CKA')
    parser.add_argument('--alpha', action = 'store', type = float, dest = 'rep_sim_alpha', default = 1.0)
    
    parser.add_argument('--logging', action = 'store', type = str, dest = 'logging', default = 'logs')
    parser.add_argument('--token_budget', action = 'store', type = int, dest = 'token_budget', default = 1e10)
    parser.add_argument('--distributed', action = 'store_true', help = 'Enable distributed training')
    
    parser.set_defaults(rep_sim = False, distributed = False, reload_dataset = False)
    args = parser.parse_args()

    return args