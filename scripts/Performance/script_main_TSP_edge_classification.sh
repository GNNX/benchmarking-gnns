#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_tsp
# tmux detach
# pkill python

# bash script_main_TSP_edge_classification.sh




############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT





############
# TSP - 4 RUNS  
############

seed0=41
seed1=42
seed2=9
seed3=23
code=main_TSP_edge_classification.py 
tmux new -s benchmark_TSP_edge_classification -d
tmux send-keys "source activate benchmark_gnn" C-m
dataset=TSP
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_MLP.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_MLP.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_MLP.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_MLP.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GCN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GCN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GCN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GCN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GIN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GIN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GIN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GIN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GraphSage.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GraphSage.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GraphSage.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GraphSage.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GatedGCN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GatedGCN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GatedGCN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GatedGCN.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GatedGCN.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GatedGCN.json' --edge_feat True &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GAT.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GAT.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GAT.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GAT.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_MoNet.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_MoNet.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_MoNet.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_MoNet.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_3WLGNN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_3WLGNN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_3WLGNN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_3WLGNN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_RingGNN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_RingGNN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_RingGNN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_RingGNN.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_TSP_edge_classification" C-m
