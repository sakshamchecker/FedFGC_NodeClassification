from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.datasets import TUDataset
torch.manual_seed(12345)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self,num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        try:
          x = x.cuda()
        except:
          pass
        try:
          edge_index = edge_index.cuda()
        except:
          pass
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# model = GCN(hidden_channels=32)
# print(model)

# model = GCN(hidden_channels=32)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()

# def train():
#     model.train()

#     for data in train_loader:  # Iterate in batches over the training dataset.
#          out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
#          loss = criterion(out, data.y)  # Compute the loss.
#          loss.backward()  # Derive gradients.
#          optimizer.step()  # Update parameters based on gradients.
#          optimizer.zero_grad()  # Clear gradients.

        

# def test(loader):
#      model.eval()

#      correct = 0
#      loss=0
#      for data in loader:  # Iterate in batches over the training/test dataset.
#          out = model(data.x, data.edge_index, data.batch)  
#          loss += criterion(out, data.y).item()
#          pred = out.argmax(dim=1)  # Use the class with highest probability.
#          correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#      return correct / len(loader.dataset), loss / len(loader.dataset)


# for epoch in range(1, 171):
#     train()
#     train_acc, train_loss = test(train_loader)
#     test_acc, test_loss = test(test_loader)
#     # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
#     #print loss and accuracy
#     print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')