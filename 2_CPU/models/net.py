import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        intermediate = []
        
        # add sequence of convolutional and max pooling layers
        x = self.conv1(x)
        intermediate.append(x.detach().numpy())
        x = self.pool(F.relu(x))
        
        x = self.conv2(x)
        intermediate.append(x.detach().numpy())
        x = self.pool(F.relu(x))
        
        x = self.conv3(x)
        intermediate.append(x.detach().numpy())
        x = self.pool(F.relu(x))
        
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        target1 = self.fc1(x)
        intermediate.append(target1.detach().numpy())
        x = F.relu(target1)
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        intermediate.append(x.detach().numpy()) 
        return target1, x, intermediate
