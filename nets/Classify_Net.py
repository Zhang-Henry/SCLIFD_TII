
# import torch.nn as nn


# class mynet(nn.Module):
#   def __init__(self,num_classes=2):
#     super(mynet, self).__init__()
#     self.encoder = nn.Sequential(
#         nn.Linear(52, 20),
#         nn.ReLU(),
#         nn.Linear(20, 10),
#         nn.ReLU()
#         )
#     self.fc =  nn.Linear(10, num_classes)


#   def forward(self, x):
#     x = self.encoder(x)
#     x = self.fc(x)
#     return x

#   def addOutputNodes(self, num_new_outputs):
#     in_features = self.fc.in_features
#     out_features = self.fc.out_features
#     weight = self.fc.weight.data

#     self.fc = nn.Linear(in_features, out_features + num_new_outputs)
#     self.fc.weight.data[:out_features] = weight





import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU, use_dropout=False, dropout_p=0.5):
        super(ResidualBlock, self).__init__()
        self.activation = activation()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_p)

        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)

        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out = out+identity
        out = self.activation(out)

        return out


class mynet(nn.Module):
    def __init__(self, num_classes=2):
        super(mynet, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(52, 64),
            nn.ReLU(),
            ResidualBlock(64, 128),
            nn.ReLU(),
            ResidualBlock(128, 256),
            nn.ReLU(),
            ResidualBlock(256, 512),
            nn.ReLU(),
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)
        # x = self.attention(x)
        x = self.fc(x)
        return x

    def addOutputNodes(self, num_new_outputs):
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features + num_new_outputs)
        self.fc.weight.data[:out_features] = weight

