import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn

# heterogeneous interaction layer
class HIL(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 dropout, graph_type='Graph_Bond', D_count=64,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_type = graph_type
        self.D_count = in_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        
        if graph_type == 'Graph_Bond':
            self.mlp_coord_cov = nn.Sequential(nn.Linear(self.D_count, self.in_channels), nn.SiLU())
            self.mlp_coord_ncov = nn.Sequential(nn.Linear(2*self.D_count, self.in_channels), nn.SiLU())
            self.mlp_neighbour = nn.Sequential(nn.Linear(2*self.in_channels, self.in_channels), nn.LeakyReLU())
#             self.mlp_neighbour = nn.Linear(2*self.in_channels, self.in_channels)
        else:
            self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
            self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

    def forward(self, init_x, x, x_bond=None, edge_index_intra=None, edge_index_inter=None, pos=None,
                size=None):

        row_cov, col_cov = edge_index_intra
        coord_diff_cov = pos[row_cov] - pos[col_cov]
        distance_cov = torch.norm(coord_diff_cov, dim=-1)
        if self.graph_type == 'Graph_Bond':
            radial_cov = self.mlp_coord_cov(_rbf(distance_cov, D_min=1., D_max=6., D_count=self.D_count, device=x.device))
            out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, edge_attr=x_bond, radial=radial_cov, size=size)
#             out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, edge_attr=None, radial=radial_cov, size=size)
        else:
            radial_cov = self.mlp_coord_cov(_rbf(distance_cov, D_min=0., D_max=6., D_count=9, GIGN=True, device=x.device))
            out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, edge_attr=None, radial=radial_cov, size=size)
#         print(x.shape, out_node_intra.shape, row_cov.shape, col_cov.shape) 
#         torch.Size([16157, 256]) torch.Size([16157, 256]) torch.Size([31722]) torch.Size([31722])

        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
        if self.graph_type == 'Graph_Bond':
            distance_ncov_colomb = 1/torch.norm(coord_diff_ncov, dim=-1)**2
            distance_ncov_london = 1/torch.norm(coord_diff_ncov, dim=-1)**6
            rbf_ncov = torch.cat((_rbf(distance_ncov_colomb, D_min=1., D_max=6., D_count=self.D_count, power=-2, device=x.device), 
                                  _rbf(distance_ncov_london, D_min=1., D_max=6., D_count=self.D_count, power=-6, device=x.device)), dim=-1)
        else:
            distance_ncov = torch.norm(coord_diff_ncov, dim=-1)
            rbf_ncov = _rbf(distance_ncov, D_min=0., D_max=6., D_count=9, GIGN=True, device=x.device)
        radial_ncov = self.mlp_coord_ncov(rbf_ncov)
        out_node_inter = self.propagate(edge_index=edge_index_inter, x=x, edge_attr=torch.empty(0, device='cuda'), radial=radial_ncov, size=size)
        out_node = self.mlp_node_cov(x + out_node_intra) + self.mlp_node_ncov(x + out_node_inter)

        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial,
                index: Tensor, edge_attr: Tensor):
        if edge_attr is not None and edge_attr.numel() > 0:
            x = self.mlp_neighbour(torch.cat((x_j, edge_attr), dim=-1)) * radial
        else:
            x = x_j * radial
        return x
    
    
def _rbf(D, D_min=1., D_max=6., D_count=16, power=1, GIGN=False, device='cuda'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    if GIGN:
        temp = D_min
        D_min = D_max**power if power < 0 else D_min**power
        D_max = temp**power if power < 0 else D_max**power
        D_sigma = (D_max - D_min) / D_count
    else:
        D_mu = D_mu**power
        D_mu_shift = torch.cat((D_mu[1:], D_mu[-2].unsqueeze(0))) if power < 0 else torch.cat((D_mu[1].unsqueeze(0), D_mu[:-1]))
        D_sigma = torch.abs(D_mu - D_mu_shift)

    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu.view([1, -1])) / D_sigma.view([1, -1])) ** 2)
    if power > 0 and not GIGN:
        RBF = torch.flip(RBF, [1])
    return RBF

# %%