import torch

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ResNetBlock, self).__init__()
        self.bn = torch.nn.BatchNorm1d(in_features)
        self.fc1 = torch.nn.Linear(in_features, out_features)
        self.fc2 = torch.nn.Linear(out_features, out_features)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        y = torch.relu(self.fc1(self.bn(x)))
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return torch.add(x, y)
    
class TabResNet(torch.nn.Module):
    def __init__(self, in_features, out_features, num_blocks=1, embedding_dim=128):
        super(TabResNet, self).__init__()
        self.embedding = torch.nn.Linear(in_features, embedding_dim)
        self.res_blocks = []
        for i in range(num_blocks):
            self.res_blocks.append(ResNetBlock(embedding_dim, embedding_dim))
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)
        self.bn = torch.nn.BatchNorm1d(embedding_dim)
        self.fc = torch.nn.Linear(embedding_dim, out_features)
        
    def forward(self, x):
        x = self.embedding(x)
        for block in self.res_blocks:
            x = block(x)
        x = torch.relu(self.bn(x))
        x = self.fc(x)
        return x
    
class TorchModel(torch.nn.Module):
    def __init__(self, model):
        super(TorchModel, self).__init__()
        self.model = model
        self.peudo_forward = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.tensor(self.model.predict_proba(x.numpy()))