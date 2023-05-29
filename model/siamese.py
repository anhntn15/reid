import torch
from torch import nn


class Siamese502(nn.Module):
    def __init__(self, img_num_channel=3):
        super(Siamese502, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(img_num_channel, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(89600, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

    def feed(self, x):
        return self.network(x)

    def forward(self, a, p, n):
        a_ = self.feed(a)
        p_ = self.feed(p)
        n_ = self.feed(n)
        return a_, p_, n_


class Siamese502SIFT(Siamese502):
    def __init__(self, nfeatures=20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(nfeatures * 128, nfeatures * 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(nfeatures * 32, nfeatures * 8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(nfeatures * 8, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )


class SiameseColorbased(Siamese502):
    def __init__(self, in_features: int, **kwargs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 50)
        )


class SiameseColorbased2(Siamese502):
    def __init__(self, color_features_1: int, color_features_2: int, last_fc=True):
        self.color_features = color_features_1
        self.last_fc = last_fc
        super().__init__()
        self.network1 = nn.Sequential(
            nn.Linear(color_features_1, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50)
        )

        self.network2 = nn.Sequential(
            nn.Linear(color_features_2, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50)
        )
        if last_fc:
            self.fc = nn.Linear(100, 50)

    def feed(self, x):
        f1 = self.network1(x[:, :self.color_features])
        f2 = self.network2(x[:, self.color_features:])
        f = torch.cat((f1, f2), dim=1)
        return self.fc(f) if self.last_fc else f


class SiameseGNN(nn.Module):
    def __init__(self, node_features, gcn_hidden_channels, fc_hidden_channels, emb_features, global_pool=True):
        """

        :param node_features: number of node features
        :param emb_features: final output dimension of model
        :param global_pool: if True, average all node features to get a graph features.
                Otherwise, directly feed all node to FC layers
        """
        super(SiameseGNN, self).__init__()
        self.global_pool = global_pool

        from torch_geometric.nn import GCNConv
        # trick to experiment with GAT instead of GCN, uncomment the below:
        # from torch_geometric.nn import GCNConv as GAT
        self.conv1 = GCNConv(node_features, gcn_hidden_channels)
        self.conv2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)
        self.conv3 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)
        self.conv4 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)
        self.conv5 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)

        self.fc1 = nn.Linear(gcn_hidden_channels, fc_hidden_channels)
        self.fc2 = nn.Linear(fc_hidden_channels, fc_hidden_channels)
        self.fc3 = nn.Linear(fc_hidden_channels, emb_features)

    def feed(self, x, edge_idx, batch):
        x = self.conv1(x, edge_idx)
        x = x.relu()
        x = self.conv2(x, edge_idx)
        x = x.relu()
        x = self.conv3(x, edge_idx)
        x = x.relu()
        x = self.conv4(x, edge_idx)
        x = x.relu()
        x = self.conv5(x, edge_idx)

        if self.global_pool:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)  # average nodes' feature -> one graph feature vector

        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)

        return x

    def forward(self, anchor, positive, negative):
        a_ = self.feed(anchor.x, anchor.edge_index, anchor.batch)
        p_ = self.feed(positive.x, positive.edge_index, positive.batch)
        n_ = self.feed(negative.x, negative.edge_index, negative.batch)

        return a_, p_, n_
