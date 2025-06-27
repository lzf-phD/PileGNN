# Self-defined loss class
# https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
import torch
import torch.nn as nn


class L1_Loss(nn.Module):  # 自定义损失函数
    def __init__(self):
        super(L1_Loss, self).__init__()

    def forward(self, edge_out, edge_y, accuracy_threshold):  #
        # Calculating loss uses normalized y
        condition = torch.abs(
            edge_y) >= accuracy_threshold
        mean_loss = torch.abs(edge_y[condition] - edge_out[condition]).sum()
        return mean_loss


class L2_Loss(nn.Module):
    def __init__(self):
        super(L2_Loss, self).__init__()

    def forward(self, edge_out, edge_y, accuracy_threshold):
        condition = torch.abs(edge_y) >= accuracy_threshold
        mse_loss = (torch.abs(edge_y[condition] - edge_out[condition]) ** 2).sum()
        return mse_loss
    



