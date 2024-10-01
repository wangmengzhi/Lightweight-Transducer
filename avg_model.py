import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch

checkpoint_avg = {}

fp='save/model.'
n_models=0
for ep in range(170,200):
    checkpoint_path = fp+str(ep)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    for k, v in checkpoint.items():
        if k not in checkpoint_avg:   
            checkpoint_avg[k]=v
        else:
            checkpoint_avg[k] += v 
    n_models += 1
    
for k, v in checkpoint_avg.items():
    checkpoint_avg[k] = (checkpoint_avg[k]/n_models).to(checkpoint_avg[k])
    
checkpoint_avg_path = fp + 'avg'
if os.path.isfile(checkpoint_avg_path):
    os.remove(checkpoint_avg_path)
torch.save(checkpoint_avg, checkpoint_avg_path)
