import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import time

"""
    Ring-GNN
    On the equivalence between graph isomorphism testing and function approximation with GNNs (Chen et al, 2019)
    https://arxiv.org/pdf/1905.12560v1.pdf
"""
from layers.ring_gnn_equiv_layer import RingGNNEquivLayer
from layers.mlp_readout_layer import MLPReadout

class RingGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.num_atom_type = net_params['num_atom_type']    # 'num_atom_type' is 'nodeclasses' as in RingGNN original repo
        self.num_bond_type = net_params['num_bond_type']
        avg_node_num = net_params['avg_node_num'] 
        radius = net_params['radius'] 
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        
        if self.edge_feat:
            self.depth = [torch.LongTensor([1+self.num_atom_type+self.num_bond_type])] + [torch.LongTensor([hidden_dim])] * n_layers
        else:
            self.depth = [torch.LongTensor([1+self.num_atom_type])] + [torch.LongTensor([hidden_dim])] * n_layers
            
        self.equi_modulelist = nn.ModuleList([RingGNNEquivLayer(self.device, m, n,
                                                                 layer_norm=self.layer_norm,
                                                                 residual=self.residual,
                                                                 dropout=dropout,
                                                                 radius=radius,
                                                                 k2_init=0.5/avg_node_num) for m, n in zip(self.depth[:-1], self.depth[1:])])
        
        self.prediction = MLPReadout(torch.sum(torch.stack(self.depth)).item(), 1) # 1 out dim since regression problem

    def forward(self, x_no_edge_feat, x_with_edge_feat):
        """
            CODE ADPATED FROM https://github.com/leichen2018/Ring-GNN/
        """
        
        x = x_no_edge_feat
        
        if self.edge_feat:
            x = x_with_edge_feat

        # this x is the tensor with all info available => adj, node feat and edge feat (if flag edge_feat true)

        x_list = [x]
        for layer in self.equi_modulelist:    
            x = layer(x)
            x_list.append(x)
        
        # # readout
        x_list = [torch.sum(torch.sum(x, dim=3), dim=2) for x in x_list]
        x_list = torch.cat(x_list, dim=1)
        
        x_out = self.prediction(x_list)

        return x_out
    
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss


