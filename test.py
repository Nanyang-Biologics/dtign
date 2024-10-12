import torch

file_path = '/home/user1/DTIGN/results/benchmark_test/20241008_130215_I1_Graph_DTIGN_Pose_1_Seed_0_hidden_dim=256_lr=0.0001_val_num=100-val_rate=1-patience=100-D_count=64/20241008_143802_GIGN_fold_4_I1_Graph_DTIGN_Pose_1_Seed_0_hidden_dim=256_lr=0.0001_val_num=100-val_rate=1-patience=100-D_count=64/result/runtime_info.pt'

data = torch.load(file_path)

# If the content is a dictionary or any object, you can inspect it
print(data)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
