import torch

    
def edge_accuracy(edge_out, edge_y, accuracy_threshold):

    device = edge_out.device
    condition = torch.abs(edge_y) >= 0
    ones = torch.ones(edge_y.shape).to(device)[condition]
    zeros = torch.zeros(edge_y.shape).to(device)[condition]
    relative_accuracy = torch.max(ones - torch.div(torch.abs(edge_y[condition] - edge_out[condition]), torch.abs(edge_y[condition])), zeros)
    return relative_accuracy.sum(), torch.numel(relative_accuracy)