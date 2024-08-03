# %%
import torch, re, time, random
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from DTIGN_HIL import HIL

def group_pocket_multi_pose(pocket_list, embedding, device):
    embedding_list, supervised_embedding_list, unique_pocket_list, current_index = [], [], [], 0
    for pockets in pocket_list:
        unique_pocket = list(set([x.split('-')[0] for x in pockets]))
        unique_pocket.sort()
        unique_pocket_list.append(unique_pocket)
        temp_embedding_list, temp_embedding_supervised = [torch.tensor([]).to(device) for x in unique_pocket], torch.tensor([]).to(device)
        for pocket in pockets:
            pocket_idx = pocket.split('-')[0]
            if pocket.startswith('G'):
                temp_embedding_supervised = torch.cat((temp_embedding_supervised, embedding[current_index].unsqueeze(0)), dim=0)
            else:
                index = unique_pocket.index(pocket_idx)
                temp_embedding_list[index] = torch.cat((temp_embedding_list[index], embedding[current_index].unsqueeze(0)), dim=0)
            current_index += 1
        embedding_list.append(temp_embedding_list)
        supervised_embedding_list.append(temp_embedding_supervised)
    
    return embedding_list, supervised_embedding_list, unique_pocket_list

def group_pocket(pocket_list, embedding, device):
    embedding_list, supervised_embedding_list, current_index = [], [], 0
    for pockets in pocket_list:
        temp_embedding, temp_embedding_supervised = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for pocket in pockets:
            if pocket.startswith('G'):
                temp_embedding_supervised = torch.cat((temp_embedding_supervised, embedding[current_index].unsqueeze(0)), dim=0)
            else:
                temp_embedding = torch.cat((temp_embedding, embedding[current_index].unsqueeze(0)), dim=0)
            current_index += 1
        embedding_list.append(temp_embedding)
        supervised_embedding_list.append(temp_embedding_supervised)
    
    return embedding_list, supervised_embedding_list


class DTIGN(nn.Module):
    def __init__(self, node_dim=35, bond_dim=10, hidden_dim=256, gconv_layer=3, fc_layer=3, task_num=1, dropout=0, attention_dropout=0, embedding_num=1, num_heads=8, num_pose=1, D_count=64, self_attention=False, graph_type='Graph_DTIGN'):
        super().__init__()
        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.gconv = gconv(gconv_layer, hidden_dim, dropout, D_count, graph_type=graph_type)
        self.fc = FC(embedding_num * hidden_dim, hidden_dim, fc_layer, dropout, task_num) # MLPPredictor or FC
        self.hidden_dim = hidden_dim
        self.num_pose = num_pose
        self.lin_bond = nn.Sequential(Linear(bond_dim, hidden_dim), nn.SiLU())
        if self_attention:
            self.self_attention_pose = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout)
            self.self_attention_pocket = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout)
            self.dropout = nn.Dropout(attention_dropout)
#             self.adaptor = nn.Sequential(Linear(hidden_dim, 64), nn.SiLU(), Linear(64, hidden_dim))
#             self.lin_pose = Linear(hidden_dim, hidden_dim)
#             self.lin_pocket = Linear(hidden_dim, hidden_dim)
#             self.lin_semi = Linear(hidden_dim, hidden_dim)
    # TODO: change to cuda later
    def forward(self, data=None, embedding = None, pocket_list=None, device = torch.device('cpu'), return_f=False, f_only=False, self_attention=False, semi_supervise=False, save_attention=False, graph_type='Graph_DTIGN'):
        if not f_only:
            init_x, edge_index_intra, edge_index_inter, pos = \
            data.x, data.edge_index_intra, data.edge_index_inter, data.pos
            x = self.lin_node(init_x)
            x_bond = None
            init_x_bond = data.x_bond
            x_bond = self.lin_bond(init_x_bond)
            x = self.gconv(init_x, x, x_bond, edge_index_intra, edge_index_inter, pos)
            embeddings = global_add_pool(x, data.batch)
            if self_attention:
#                 embedding_list, supervised_embedding_list, unique_pocket_list = group_pocket_multi_pose(pocket_list, embeddings, device)
                embedding_list, supervised_embedding_list = group_pocket(pocket_list, embeddings, device)
                embeddings, attn_output_weights, pose_attn_output_weights, pocket_supervised_loss, loss_counter = torch.tensor([]).to(device), [], [], 0, 0
                for i, embedding in enumerate(embedding_list):
                    attn_output, attn_output_weight = self.self_attention_pocket(embedding, embedding, embedding) # (7,256) (7,7)
                    attn_output_weights.append(attn_output_weight)
                    residual_output = self.dropout(attn_output) + embedding
                    residual_output = residual_output.sum(dim=0, keepdim=True)
                    temp_embeddings = residual_output
#                     temp_embeddings = self.lin_pocket(residual_output)
                    supervised_embedding = supervised_embedding_list[i]
                    if supervised_embedding.numel() != 0 and semi_supervise:
                        reduced_supervised_embedding = supervised_embedding.sum(dim=0, keepdim=True)
                        pocket_supervised_loss += 1 - F.cosine_similarity(temp_embeddings, reduced_supervised_embedding, dim=1)
                        temp_embeddings = torch.cat((temp_embeddings, supervised_embedding), dim=0)
                        loss_counter += 1
                    embeddings = torch.cat((embeddings, temp_embeddings), dim=0)
                pocket_supervised_loss = pocket_supervised_loss/(loss_counter+1e-9)
                if return_f:
                    return embeddings
                x = self.fc(embeddings)
                if save_attention:
                    return x.view(-1), [pocket_supervised_loss, attn_output_weights, pose_attn_output_weights]
                return x.view(-1), [pocket_supervised_loss]
            
        if return_f:
            return embeddings
        x = self.fc(embeddings)
        return x.view(-1)


class gconv(nn.Module):   
    def __init__(self, gconv_layer, hidden_dim, dropout, D_count, graph_type='Graph_DTIGN'):
        super(gconv, self).__init__()
        self.gconv_layer = gconv_layer
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gconv = nn.ModuleList()
        for i in range(self.gconv_layer):
            self.gconv.append(HIL(hidden_dim, hidden_dim, dropout, D_count, graph_type))
            
    def forward(self, init_x, x, x_bond=None, edge_index_intra=None, edge_index_inter=None, pos=None):
        for layer in self.gconv:
            x = layer(init_x=init_x, x=x, x_bond=x_bond, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, pos=pos)
        return x
    
    
class MLPPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_FC_layer=2, dropout=0., n_tasks=1):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, feats):
        return self.predict(feats)

    
class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        # self.dropout = 0.1
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h

# %%