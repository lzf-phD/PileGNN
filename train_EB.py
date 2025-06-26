import torch
from torch_geometric.data import DataLoader
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from dataset.dataset_EB import PileDataset
from GNN.models import *
from GNN import losses
from utils import plot
from utils import datasets
from utils import accuracy

import time
from datetime import datetime
import os
import json
import random

space = plot.print_space() #画分割线
# Args
parser = ArgumentParser()
# general args
# Dataset
parser.add_argument('--dataset_name', dest='dataset_name', default='Pile_EB', type=str)
parser.add_argument('--data_num', dest='data_num', default=4720, type=int)
parser.add_argument('--train_ratio', dest='train_ratio', default=0.9, type=float)
parser.add_argument('--normalization', dest='normalization', default=True, type=bool)

# GNN
parser.add_argument('--model', dest='model', default='Pile_GraphNetwork', type=str, help='model class name. E.g., GCN, Tony_GNN, ...')
parser.add_argument('--aggr_type', dest='aggr', default='mean', type=str)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=32, type=int)
parser.add_argument('--dropout_p', dest='dropout_p', default=0.1, type=float)
parser.add_argument('--gnn_dropout', dest='gnn_dropout', default='True', type=bool, help='whether dropout, default 0.5')
parser.add_argument('--gnn_act', dest='gnn_act', default='True', type=bool, help='whether activation, default RELU')

# Training
parser.add_argument('--target', dest='target', default='all', type=str, help='which output target you are going to train, like displacement, moement, shear......')
parser.add_argument('--epoch_num', dest='epoch_num', default=100, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
parser.add_argument('--lr', dest='lr', default=5e-5, type=float)
parser.add_argument('--loss_function', dest='loss_function', default='L1_Loss', type=str)
parser.add_argument('--accuracy_threshold', dest='accuracy_threshold', default=0, type=float, help='The normalized output value smaller than the threshold will be ignored.')

# Training
parser.add_argument('--training_time', dest='training_time', default=0, type=float)

args = parser.parse_args()
print(args, end=space)
date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
print("Start time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)

# Load in data
dataset = PileDataset(split='train')
print(f"dataset: {args.dataset_name}")

# Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", end=space)

# Split data into train, test set
train_dataset, valid_dataset, _ = datasets.split_dataset(dataset,train_ratio=args.train_ratio,valid_ratio=1-args.train_ratio)
train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=1,shuffle=True)
# Model setup
input_dim = 1
edge_attr_dim = 6
edge_output_dim =1
model_constructor_args = {
    'input_dim': input_dim, 'hidden_dim': args.hidden_dim, 'dropout_p': args.dropout_p,
    'aggr': args.aggr, 'edge_attr_dim': edge_attr_dim,
    'gnn_act': args.gnn_act, 'gnn_dropout': args.gnn_dropout, 'device': device
} # {input_dim=1,hidden_dim=256，hidden_dim=0，aggr=mean，edge_attr_dim=4，gnn_act=True，gnn_dropout=True，device=cuda0}

model = locals()[args.model](**model_constructor_args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.loss_function == 'L1_Loss':
    criterion = nn.L1Loss()

accuracy_record = np.zeros((3, args.epoch_num))
loss_record = np.zeros((3, args.epoch_num))
best_accuracy = 0

# Save Model path
task = args.dataset_name
save_model_dir = 'checkpoints/'+task+'/'
if os.path.exists(save_model_dir) == False:
    os.makedirs(save_model_dir)

# Training phase
start_time = time.time()
for epoch in range(1,args.epoch_num+1):
    # Train
    model.train()
    for data,_,_ in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        edge_out = model(data.x, data.edge_index, data.edge_attr,data.P,data.D,data.K)
        loss = criterion(edge_out, data.y)
        loss.backward()
        optimizer.step()
    # Get train and test score
    model.eval()
    with torch.no_grad():
         for i, loader in enumerate([train_loader, valid_loader]): #列表有两个训练集，依次加载
             loss_val = 0
             total_correct = 0
             total_elems = 0
             for data,_,_ in loader:
                 data = data.to(device)
                 edge_out = model(data.x, data.edge_index, data.edge_attr,data.P,data.D,data.K)
                 loss = criterion(edge_out, data.y)
                 loss_val += loss.item()
                 correct, elems = accuracy.edge_accuracy(edge_out,data.y,0)
                 total_correct += correct
                 total_elems += 1

             accuracy_record[i][epoch] = total_correct / total_elems
             loss_record[i][epoch] = loss_val / total_elems

    print(f' Epoch: {epoch:03d},  Train Acc: {accuracy_record[0][epoch]:.4f},  Valid Acc: {accuracy_record[1][epoch]:.4f},  ' +
         f'Train Loss: {loss_record[0][epoch]:.4f},  Valid Loss: {loss_record[1][epoch]:.4f}')
    torch.save(model.state_dict(), save_model_dir + f'model_{epoch}.pth')
    print(f'{epoch}.epoch Trained model saved')

finish_time = time.time()
args.training_time = (finish_time - start_time)/60
print(space)
print(f"Time spent: {(finish_time - start_time)/60:.2f} min")
print("Finish time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)

with open(save_model_dir + 'training_args.json', 'w') as f:
    json.dump(vars(args), f)

# Plot the results
# visualize_graph(dataset[int(torch.rand(1)*50)], save_model_dir)
plot.plot_learningCurve(accuracy_record, save_model_dir, title=', '.join([args.model, task+'\n', date_str, args.target]), target=args.target)
plot.plot_lossCurve(loss_record, save_model_dir, title=', '.join([args.model, task+'\n', date_str, args.target]), target=args.target)
