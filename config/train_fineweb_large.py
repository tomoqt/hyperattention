

wandb_log = True
wandb_project = 'fineweb-tensor_attention'
wandb_run_name='tensor_attention_large'

batch_size = 24
block_size = 1024
gradient_accumulation_steps =  1
dataset = 'fineweb'
max_iters = 60000
lr_decay_iters = 60000
n_layer = 8
n_head = 6
n_embd = 1024
dropout = 0.0
# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
use_muon = False
muon_lr = 1e-3
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5

#hyperattention
higher_order_mode = 'sequential'
