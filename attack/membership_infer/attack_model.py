import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # define nn
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_classes, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        # print("attack X",X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = self.softmax(X)

        return X