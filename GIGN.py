# %%
import torch, re, time, random
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from HIL import HIL

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


class GIGN(nn.Module):
    def __init__(self, node_dim=35, bond_dim=10, hidden_dim=256, gconv_layer=3, fc_layer=3, task_num=1, dropout=0, attention_dropout=0, embedding_num=1, num_heads=8, num_pose=1, self_attention=False, graph_type='Graph_Bond'):
        super().__init__()
        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.gconv = gconv(gconv_layer, hidden_dim, dropout, graph_type=graph_type)
        self.fc = FC(embedding_num * hidden_dim, hidden_dim, fc_layer, dropout, task_num)
        self.hidden_dim = hidden_dim
        self.num_pose = num_pose
        if graph_type == 'Graph_Bond':
            self.lin_bond = nn.Sequential(Linear(bond_dim, hidden_dim), nn.SiLU())
        if self_attention:
            self.self_attention_pose = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout)
            self.self_attention_pocket = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout)
            self.dropout = nn.Dropout(attention_dropout)
#             self.lin_pose = Linear(hidden_dim, hidden_dim)
#             self.lin_pocket = Linear(hidden_dim, hidden_dim)
#             self.lin_semi = Linear(hidden_dim, hidden_dim)

    def forward(self, data=None, embedding = None, pocket_list=None, device = torch.device('cuda'), return_f=False, f_only=False, self_attention=False, semi_supervise=False, save_attention=False, graph_type='Graph_Bond'):
        if not f_only:
            init_x, edge_index_intra, edge_index_inter, pos = \
            data.x, data.edge_index_intra, data.edge_index_inter, data.pos
            x = self.lin_node(init_x)
            x_bond = None
            if graph_type == 'Graph_Bond':
                init_x_bond = data.x_bond
                x_bond = self.lin_bond(init_x_bond)
            x = self.gconv(init_x, x, x_bond, edge_index_intra, edge_index_inter, pos)
            embeddings = global_add_pool(x, data.batch)
            if self_attention:
#                 embedding_list, supervised_embedding_list, unique_pocket_list = group_pocket_multi_pose(pocket_list, embeddings, device)
                embedding_list, supervised_embedding_list = group_pocket(pocket_list, embeddings, device)
                embeddings, attn_output_weights, pose_attn_output_weights, pocket_supervised_loss, loss_counter = torch.tensor([]).to(device), [], [], 0, 0
                for i, embedding in enumerate(embedding_list):
#                     if self.num_pose > 2:
#                         pocket_embedding, pose_attn_output_weight = torch.tensor([]).to(device), []
#                         for poses in embedding:
# #                             print('pose num:', poses.shape[0])
#                             attn_output_pose, attn_output_weight_pose = self.self_attention_pose(poses, poses, poses)
#                             residual_output = self.dropout(attn_output_pose) + poses
# #                             residual_output = self.layer_norm(residual_output)
#                             readout_output = self.lin_pose(residual_output.sum(0, keepdim=True))
#                             pocket_embedding = torch.cat((pocket_embedding, readout_output), dim=0)
#                             pose_attn_output_weight.append(attn_output_weight_pose)
#                         embedding = pocket_embedding
#                         pose_attn_output_weights.append(pose_attn_output_weight)
#                     elif self.num_pose == 2:
#                         pocket_embedding, pose_attn_output_weight = torch.tensor([]).to(device), []
#                         for poses in embedding:
#                             current_time = time.time()
#                             random_seed = int(current_time * 1e6)
#                             rng = random.Random(random_seed)
#                             random_index = rng.randint(0, poses.shape[0] - 1)
#                             pocket_embedding = torch.cat((pocket_embedding, poses[random_index].unsqueeze(0)), dim=0)
# #                             pocket_embedding = torch.cat((pocket_embedding, poses.sum(0, keepdim=True)), dim=0)
#                         embedding = pocket_embedding
#                     else:
#                         embedding = torch.cat(embedding)
#                     if len(unique_pocket_list[i]) > 2:
#                         print('pocket num:', len(unique_pocket_list[i]))
                    attn_output, attn_output_weight = self.self_attention_pocket(embedding, embedding, embedding) # (7,256) (7,7)
                    attn_output_weights.append(attn_output_weight)
                    residual_output = self.dropout(attn_output) + embedding
                    residual_output = residual_output.sum(dim=0, keepdim=True)
                    temp_embeddings = residual_output
#                     temp_embeddings = self.lin_pocket(residual_output)
#                     else:
#                         temp_embeddings = embedding.sum(dim=0, keepdim=True)
#                         attn_output_weights.append([])
                    supervised_embedding = supervised_embedding_list[i]
                    if supervised_embedding.numel() != 0 and semi_supervise:
#                         sup_len = supervised_embedding.shape[0]
#                         temp_embeddings_expanded = temp_embeddings.expand(sup_len, -1)
#                         supervised_embedding = self.lin_pocket(supervised_embedding)
#                         pocket_supervised_loss += torch.norm(temp_embeddings_expanded - supervised_embedding, p=2, dim=-1)
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
    def __init__(self, gconv_layer, hidden_dim, dropout, graph_type='Graph_Bond'):
        super(gconv, self).__init__()
        self.gconv_layer = gconv_layer
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gconv = nn.ModuleList()
        for i in range(self.gconv_layer):
            self.gconv.append(HIL(hidden_dim, hidden_dim, dropout, graph_type=graph_type))
            
    def forward(self, init_x, x, x_bond=None, edge_index_intra=None, edge_index_inter=None, pos=None):
        for layer in self.gconv:
            x = layer(init_x=init_x, x=x, x_bond=x_bond, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, pos=pos)
        return x
    
                
class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
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