import time
import dgl
import torch
from torch.utils.data import Dataset

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator


class COLLABDataset(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglLinkPropPredDataset(name='ogbl-collab')
        
        self.graph = self.dataset[0]  # single DGL graph
        
        # Create edge feat by concatenating weight and year
        self.graph.edata['feat'] = torch.cat( 
            [self.graph.edata['edge_weight'], self.graph.edata['edge_year']], 
            dim=1 
        )
        
        self.split_edge = self.dataset.get_edge_split()
        self.train_edges = self.split_edge['train']['edge']  # positive train edges
        self.val_edges = self.split_edge['valid']['edge']  # positive val edges
        self.val_edges_neg = self.split_edge['valid']['edge_neg']  # negative val edges
        self.test_edges = self.split_edge['test']['edge']  # positive test edges
        self.test_edges_neg = self.split_edge['test']['edge_neg']  # negative test edges
        
        self.evaluator = Evaluator(name='ogbl-collab')
        
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))
