import os
from argparse import ArgumentParser
from datetime import datetime

from torch_geometric.data import DataLoader
from tqdm import tqdm

from GNN.models import *
from dataset.dataset_EB import PileDataset
from utils import datasets
from utils import plot

space = plot.print_space() #画分割线
# Args
parser = ArgumentParser()
# general args
# Dataset
parser.add_argument('--dataset_name', dest='dataset_name', default='Pile_EB', type=str)
parser.add_argument('--data_num', dest='data_num', default=472, type=int)
parser.add_argument('--train_ratio', dest='train_ratio', default=1.0, type=float)
parser.add_argument('--normalization', dest='normalization', default=True, type=bool)

# GNN
parser.add_argument('--model', dest='model', default='Pile_GraphNetwork', type=str, help='model class name. E.g., GCN, Tony_GNN, ...')
parser.add_argument('--aggr_type', dest='aggr', default='mean', type=str)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=32, type=int)
parser.add_argument('--dropout_p', dest='dropout_p', default=0.0, type=float)
parser.add_argument('--gnn_dropout', dest='gnn_dropout', default='True', type=bool, help='whether dropout, default 0.5')
parser.add_argument('--gnn_act', dest='gnn_act', default='True', type=bool, help='whether activation, default RELU')

# Training
parser.add_argument('--target', dest='target', default='all', type=str, help='which output target you are going to train, like displacement, moement, shear......')
parser.add_argument('--epoch_num', dest='epoch_num', default=1000, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
parser.add_argument('--lr', dest='lr', default=5e-5, type=float)
parser.add_argument('--loss_function', dest='loss_function', default='L1_Loss', type=str)
parser.add_argument('--accuracy_threshold', dest='accuracy_threshold', default=1e-4, type=float, help='The normalized output value smaller than the threshold will be ignored.')

# Training
parser.add_argument('--training_time', dest='training_time', default=0, type=float)

args = parser.parse_args()

date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
print("Start time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)

# Load in data
dataset = PileDataset(split='test')
# Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Split data into train, test set
test_dataset, valid_dataset, _ = datasets.split_dataset(dataset,train_ratio=1,valid_ratio=0)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

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
model.eval()
model.load_state_dict(torch.load('checkpoints/Pile_EB/model_100.pth', map_location=device))
# Save result path
task = args.dataset_name

output_dir = "result\pile_EB"
os.makedirs(output_dir, exist_ok=True)

for data,filename,H in tqdm(test_loader):
    data = data.to(device)
    H_list = [t.item() for t in H]
    edge_out = model(data.x, data.edge_index, data.edge_attr,data.P,data.D,data.K)
    pred = edge_out.squeeze().tolist() # penetration ratio
    for i in range(len(pred)):
        if pred[i] > 0.95:
            pred[i] = 1.0
        elif pred[i] < 0.05:
            pred[i] = 0.0
        else:
            pred[i] = round(pred[i], 2)
    txt_name = filename[0]
    PL = round(sum(a * b for a, b in zip(pred, H_list)), 2)  # pile length
    save_path = os.path.join(output_dir, f"{txt_name}.txt")
    with open(save_path, "w") as f:
        for p in pred:
            f.write(str(p) + "\n")
    print(pred)
    print(PL)






