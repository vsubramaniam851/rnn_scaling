# Guided RNN command

torchrun --nproc_per_node=2 -m rnn_scaling.language_model \
    --exp_name rnn_gpt2_medium_pretrain \
    --student_model RNN \
    --batch_size 128 \
    --context_length 200
    --num_workers 24 \
    --hidden_dim 4200 \
    --embedding_dim 4200 \
    --num_layers 3 \
    --lr 1e-4 \
    --distributed \
    --rep_sim \
    --repdist CKA \
    --d_model 768 \
    --trans_layers 6 \
    --nheads 16 \
    --d_fd 3072 \
    --reload_dataset \
    --seed 0

# Plain RNN command

torchrun --nproc_per_node=2 -m rnn_scaling.language_model \
    --exp_name rnn_medium_pretrain \
    --student_model RNN \
    --batch_size 128 \
    --num_workers 24 \
    --hidden_dim 4200 \
    --embedding_dim 4200 \
    --num_layers 3 \
    --lr 1e-4 \
    --distributed \
    --seed 0

# Transformer Command

torchrun --nproc_per_node=2 -m rnn_scaling.language_model \
    --exp_name gpt2_medium_pretrain \
    --student_model RNN \
    --batch_size 128 \
    --num_workers 24 \
    --d_model 1024 \
    --trans_layers 24 \
    --nheads 16 \
    --d_fd 3072 \
    --lr 1e-4 \
    --distributed \
    --seed 0
