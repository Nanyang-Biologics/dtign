from fastapi import FastAPI, HTTPException
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy.stats
from typing import List
from DTIGN import DTIGN
from utils import *
from models import InferenceInput
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = 256
num_pose = 1
dropout = 0
graph_type = ['Graph_GIGN', 'Graph_DTIGN'][1]
D_count = 16
semi_supervise = False
app = FastAPI()

def load_model(ckpt_path: str):
    model = DTIGN(node_dim=35, bond_dim=10, hidden_dim=hidden_dim, num_pose=num_pose, dropout=dropout, 
                  self_attention=True, graph_type=graph_type, D_count=D_count).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model

def reduction_by_pocket(tensor, pocket_list, device, semi_supervise=False):
    output_tensor, current_index, tensor_shape = torch.tensor([]).to(device), 0, tensor.shape[0]
    for pockets in pocket_list:
        additional_num = 0
        for pocket in pockets:
            if pocket.startswith('G'):
                additional_num += 1
        pocket_num = len(pockets)
        reducted_tensor = tensor[current_index:current_index+pocket_num].mean(dim=0, keepdim=True)
        output_tensor = torch.cat((output_tensor, reducted_tensor), dim=0)
        if semi_supervise:
            for i in range(additional_num):
                output_tensor = torch.cat((output_tensor, reducted_tensor), dim=0)
        current_index += pocket_num
    
    return output_tensor

@app.get("/health")
def check_health():
    return { 'health': 'ok' }

@app.post("/inference")
def inference(request: InferenceInput):
    model = load_model("/home/user1/DTIGN/results/benchmark_test/20241014_063447_I3_Graph_DTIGN_Pose_3_Seed_0_hidden_dim=128_lr=0.0001_val_num=50-val_rate=1-patience=60-D_count=16/20241014_171319_GIGN_fold_2_I3_Graph_DTIGN_Pose_3_Seed_0_hidden_dim=128_lr=0.0001_val_num=50-val_rate=1-patience=60-D_count=16/model/Best, epoch-127, mean-5.8399, std-1.6963, train_loss-1.0610, train_rmse-1.0300, valid_rmse-0.9909, valid_pr-0.8563, valid_tau-0.6389, test_rmse-1.6809, test_pr-0.3986, test_tau-0.2521.pt")  # Path to model checkpoint

    predict_list, label_list = [], []
    predict_dict, label_dict = {}, {}

    match = re.search(r"mean-([0-9.]+), std-([0-9.]+)", file_path)
    if match:
        mean_value = float(match.group(1))
        std_value = float(match.group(2))
        print(f"Mean: {mean_value}, Std: {std_value}")
    else:
        print("Mean and Std not found.")

    # Process each data item in the request
    for item in request.data:
        # Convert data to PyTorch tensors and move to device (GPU/CPU)
        label = torch.tensor(item.y, dtype=torch.float32).to(device)
        pocket = torch.tensor(item.pocket, dtype=torch.float32).to(device)
        idxes = torch.tensor(item.idx, dtype=torch.long).to(device)
        
        with torch.no_grad():
            pred, outputs = model(data=label, pocket_list=pocket, self_attention=True, semi_supervise=False, 
                                  save_attention=False, graph_type=graph_type)
            pocket_supervised_loss = outputs[0]
            pred = standardize(pred, mean_value, std_value, device, reverse=True)
            label = reduction_by_pocket(label, pocket, device, semi_supervise=False)
            idx_list = idxes.cpu().numpy().tolist()

            # Store predictions and labels for each idx
            for i, idx in enumerate(idx_list):
                predict_dict[idx] = [] if idx not in predict_dict else predict_dict[idx]
                label_dict[idx] = [] if idx not in label_dict else label_dict[idx]
                predict_dict[idx].append(pred[i].detach().cpu().item())
                label_dict[idx].append(label[i].detach().cpu().item())

    # Summarize predictions and labels
    for key, pred in predict_dict.items():
        mean_pred = sum(pred) / len(pred)
        predict_list.append(mean_pred)
        label_item = label_dict[key]
        mean_label = sum(label_item) / len(label_item)
        label_list.append(mean_label)
    
    # Convert to numpy arrays for calculation
    pred = np.array(predict_list)
    label = np.array(label_list)

    # Return RMSE, Correlation Coefficient, and Tau as response
    rmse = np.sqrt(mean_squared_error(label, pred))
    corr_coefficient = np.corrcoef(pred, label)[0, 1]
    tau, _ = scipy.stats.kendalltau(pred, label)

if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="localhost", port=5000)