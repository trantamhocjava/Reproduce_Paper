_base_ = './base.py'
clip_model = 'ViT-L/14'
dataset = 'CUB'
data_root = f'datasets/{dataset}'
n_shots = 'all'
exp_root = f'exp/LProbe/{dataset}/{clip_model.replace("/", "_")}'

num_class = 200
