import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn=torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU()))
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn=torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU())))
        self.lin1 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)  # Global pooling layer
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return x
