from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.datasets import TUDataset
torch.manual_seed(12345)
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv


class Sage(torch.nn.Module):
  def __init__(self,num_node_features, num_classes):
    super().__init__()
    self.conv1=SAGEConv(num_node_features, 256)
    self.conv2=SAGEConv(256, num_classes)
  def forward(self,x,edge_index):
    x=self.conv1(x, edge_index)
    x=F.relu(x)
    x=F.dropout(x, training=self.training)
    x=self.conv2(x, edge_index)

    return F.log_softmax(x,dim=1)
