"""
    Utility functions for training one epoch 
    and evaluating one epoch, for GNN with dense
    tensors such as RingGNN and 3WLGNN
"""
import torch
import torch.nn as nn
import math

from train.metrics import accuracy_TU
from train.metrics import accuracy_MNIST_CIFAR
from train.metrics import MAE
from train.metrics import accuracy_SBM
from train.metrics import binary_f1_score

"""
    For TUs dataset 
"""

def train_epoch_TUs(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)
        
        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels) 
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy_TU(scores, labels)
        nb_data += labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network_TUs(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy_TU(scores, labels)
            nb_data += labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc




"""
    For superpixels datasets
"""


def train_epoch_superpixels(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)
        
        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels) 
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy_MNIST_CIFAR(scores, labels)
        nb_data += labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network_superpixels(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy_MNIST_CIFAR(scores, labels)
            nb_data += labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc





"""
    For CSL dataset
"""


def train_epoch_CSL(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)
        
        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels) 
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy_MNIST_CIFAR(scores, labels)
        nb_data += labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network_CSL(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy_MNIST_CIFAR(scores, labels)
            nb_data += labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc




"""
    For molecules (ZINC) dataset
"""

def train_epoch_molecules(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_no_edge_feat, x_with_edge_feat, targets) in enumerate(data_loader):
        if x_no_edge_feat is not None:
            x_no_edge_feat = x_no_edge_feat.to(device)
        if x_with_edge_feat is not None:
            x_with_edge_feat = x_with_edge_feat.to(device)
        targets = targets.to(device)
        
        scores = model.forward(x_no_edge_feat, x_with_edge_feat)
        loss = model.loss(scores, targets)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(scores, targets)
        nb_data += targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_molecules(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_no_edge_feat, x_with_edge_feat, targets) in enumerate(data_loader):
            if x_no_edge_feat is not None:
                x_no_edge_feat = x_no_edge_feat.to(device)
            if x_with_edge_feat is not None:
                x_with_edge_feat = x_with_edge_feat.to(device)
            targets = targets.to(device)
            
            scores = model.forward(x_no_edge_feat, x_with_edge_feat)
            loss = model.loss(scores, targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(scores, targets)
            nb_data += targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae





"""
    For SBMs datasets
"""

def train_epoch_SBMs(model, optimizer, device, data_loader, epoch, batch_size):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)

        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy_SBM(scores, labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer



def evaluate_network_SBMs(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy_SBM(scores, labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc




"""
    For TSP dataset
"""

def train_epoch_TSP(model, optimizer, device, data_loader, epoch, batch_size):

    model.train()
    epoch_loss = 0
    epoch_train_f1 = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_no_edge_feat, x_with_edge_feat, labels, edge_list) in enumerate(data_loader):
        if x_no_edge_feat is not None:
            x_no_edge_feat = x_no_edge_feat.to(device)
        if x_with_edge_feat is not None:
            x_with_edge_feat = x_with_edge_feat.to(device)
        labels = labels.to(device)
        edge_list = edge_list[0].to(device), edge_list[1].to(device)
        
        scores = model.forward(x_no_edge_feat, x_with_edge_feat, edge_list)
        loss = model.loss(scores, labels)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.detach().item()
        epoch_train_f1 += binary_f1_score(scores, labels)
    epoch_loss /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    
    return epoch_loss, epoch_train_f1, optimizer


def evaluate_network_TSP(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_f1 = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_no_edge_feat, x_with_edge_feat, labels, edge_list) in enumerate(data_loader):
            if x_no_edge_feat is not None:
                x_no_edge_feat = x_no_edge_feat.to(device)
            if x_with_edge_feat is not None:
                x_with_edge_feat = x_with_edge_feat.to(device)
            labels = labels.to(device)
            edge_list = edge_list[0].to(device), edge_list[1].to(device)

            scores = model.forward(x_no_edge_feat, x_with_edge_feat, edge_list)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_f1 += binary_f1_score(scores, labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
        
    return epoch_test_loss, epoch_test_f1

