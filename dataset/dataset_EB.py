import glob
import os
import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import InterpolationMode
import numpy as np
import re
from torch_geometric.data import Data

class PileDataset(Dataset):

    def __init__(self, split,txt_path = 'data/data_EB'):
        self.split = split  # train
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.source_graph,self.label_graph,self.para_graph = self.load_graphs(txt_path)

    def load_graphs(self, txt_path):

        source_txt = glob.glob(os.path.join(txt_path, f"{self.split}_A", "*.txt"))
        target_txt = glob.glob(os.path.join(txt_path, f"{self.split}_B", "*.txt"))
        parameter_txt = glob.glob(os.path.join(txt_path, f"cond/{self.split}", "*.txt"))

        source_txt = sorted(source_txt)
        target_txt = sorted(target_txt)
        parameter_txt = sorted(parameter_txt)

        return source_txt,target_txt,parameter_txt

    def __len__(self):
        return len(self.source_graph)

    def __getitem__(self, index):
        source_path = self.source_graph[index]
        label_path = self.label_graph[index]
        txt_path = self.para_graph[index]
        with open(source_path, 'r', encoding='utf-8') as f:
            source_lines = [line.strip() for line in f if line.strip()]
        label_lines = np.loadtxt(label_path, dtype=str, encoding="utf-8").tolist()
        txt_lines = np.loadtxt(txt_path, dtype=str, encoding="utf-8").tolist()
        edge_attr,D,P,K,edge_num,H_list = get_trans_mc(txt_lines,source_lines)
        x, edge_index = get_trans_source(source_lines,edge_num)
        y = get_trans_label(label_lines)
        node_num = edge_num + 1
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, node_num=node_num, edge_num=edge_num, D=D,
                    P=P, K=round(K, 2))

        return data,os.path.basename(self.source_graph[index]),H_list

def get_trans_mc(txt_lines,source_lines):
    txt1, txt2, txt3, txt4, txt5, txt6, txt7 = [], [], [], [], [], [], []
    txt_group = [txt1, txt2, txt3, txt4, txt5, txt6, txt7]
    for i, line in enumerate(txt_lines):
        txt_group[i].extend(map(float, line.split(",")))
    node_elevation = []
    for line in source_lines:
        if "node" in line:
            node_str = re.findall(r'-?\d+\.?\d*', line)
            node_xyz = [float(x) for x in node_str]  # 坐标是浮点数类型
            node_elevation.append(node_xyz[-1])
    D = 0; P = 0
    if txt5[0] == 1.2:
        D = 0.2
        P = 0.2
    elif txt5[0] == 1.5:
        D = 0.3
        P = 0.3
    elif txt5[0] == 1.8:
        D = 0.4
        P = 0.4
    elif txt5[0] == 2.0:
        D = 0.5
        P = 0.5
    elif txt5[0] == 2.5:
        D = 0.6
        P = 0.6
    elif txt5[0] == 1.0:
        D = 0.1
        P = 0.1
    K = normalize_parameter(txt7[0],2.0,3.0)
    edge_attr = torch.zeros([len(txt1), 6])
    H_list = []
    for z in range(len(txt1)):
        hi = abs(node_elevation[z] - node_elevation[z + 1])
        H_list.append(hi)
        edge_attr[z, 0] = K
        edge_attr[z, 1] = D
        edge_attr[z, 2] = round(normalize_parameter(hi,0,15),2) #l
        edge_attr[z, 3] = 0.05 * normalize_parameter(txt1[z], 0, 25)  # gama
        edge_attr[z, 4] = 0.05 * normalize_parameter(txt2[z], 0, 1500)  # fa0
        edge_attr[z, 5] = 0.9 * normalize_parameter(txt4[z], 0, 40)  # frk
    return edge_attr,D,P,K,len(txt1),H_list

def get_trans_source(source_lines,edge_num):
    node_list = []
    for i, line in enumerate(source_lines):
        if "node" in line:
            node_str = re.findall(r'-?\d+\.?\d*', line)
            node_xyz = [float(x) for x in node_str]
            node_list.append(-node_xyz[-1])
    node_list = [normalize_parameter(node, 0, 85)for node in node_list]
    x = torch.tensor(node_list, dtype=torch.float32).unsqueeze(1)
    edge_index = torch.zeros([2, edge_num], dtype=torch.long)
    edge_index[0] = torch.arange(0, edge_num)
    edge_index[1] = torch.arange(1, edge_num+1)
    return x,edge_index

def get_trans_label(label_lines):
    txt = list(map(float, label_lines.split(',')))
    y = torch.tensor(txt, dtype=torch.float).unsqueeze(1)
    return y

def normalize_parameter(x, min_val, max_val):
    return 2*(x - min_val) / (max_val - min_val)-1

