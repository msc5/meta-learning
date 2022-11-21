import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary


# Implementation of Relation Networks described in the paper: Learning to Compare: Relation Network for Few-Shot Learning


class Embedding(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Arguments:
            in_channels: Number of channels in input
            out_channels: Number of filters in a convolution layer
        """
        super(Embedding, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, is_sup_set: bool):
        num_classes, num_examples_per_class, chan, height, width = x.shape
        z = x.view(-1, chan, height, width)
        # print(z.shape)
        x1 = self.conv1(z)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        if is_sup_set:
            # reshape back into num_class x num_examples x [determined by convolution filters and downsampling]

            # Omniglot
            x4 = x4.view(num_classes, num_examples_per_class, 64, 5, 5)

            # Mini Image net
            # x4 = x4.view(num_classes, num_examples_per_class, 64, 19, 19)

            # element-wise sum of embeddings of all examples of each class
            x4 = torch.sum(x4, 1)
        return x4


class Relation(nn.Module):

    def __init__(self, in_channels, out_channels, in_features):
        """
        Arguments:
            in_channels: Number of channels in input
            out_channels: Number of filters in a convolution layer
            in_features: Length of input into first FC layer
        """
        super(Relation, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # No padding here for ImageNet
        # Padding="same" for Omniglot
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # need to reshape again like above for convolution to work
        _, _, map_chan, height, width = x.shape
        z = x.view(-1, map_chan, height, width)
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.flatten(z)
        z = self.linear1(z)
        z = self.linear2(z)
        return z


class RelationNetwork(nn.Module):

    def __init__(self, in_embed, out_embed, in_feat_rel, num_classes, support_num_examples_per_class, query_num_examples_per_class):
        """
        Relation Network
        Arguments:
        """
        super(RelationNetwork, self).__init__()
        self.num_classes = num_classes
        self.support_num_examples_per_class = support_num_examples_per_class
        self.query_num_examples_per_class = query_num_examples_per_class
        self.__name__ = 'RelationNetwork'

        self.embed = Embedding(in_embed, out_embed)
        # 128, 64
        self.relation = Relation(2 * out_embed, out_embed, in_feat_rel)

    def forward(self, support_set, query_set):
        query_embed = self.embed(query_set, False)
        support_embed = self.embed(support_set, True)
        # print(f'query_embed shape: {query_embed.shape}')
        # print(query_embed)
        # print()
        # print(f'support_embed shape: {support_embed.shape}')
        # print(support_embed)
        # print()

        # concat every query with every class
        # thus, in the support set, each class will only have one embedding (num_class * 1 for query set) whereas in the query set, each class will have query_num_examples_per_class embeddings (num_class * query_num_examples_per_class for supp set)
        query_embed = query_embed.repeat(self.num_classes * 1, 1, 1, 1, 1)
        query_embed = torch.permute(query_embed, (1, 0, 2, 3, 4))
        support_embed = support_embed.repeat(
            self.num_classes * self.query_num_examples_per_class, 1, 1, 1, 1)
        # print(f'query_embed shape: {query_embed.shape}')
        # print()
        # print(f'support_embed shape: {support_embed.shape}')
        # print()

        # concat along depth (num_filters) which is in dim=2
        feature_map = torch.cat((query_embed, support_embed), 2)
        # print(f'feature_map shape {feature_map.shape}')
        # print(feature_map)
        # print()
        # score is a 2D array of shape (num_examples_in query_set x num_classes)
        # Each row contains all the relation scores for a query example
        score = self.relation(feature_map)
        score = score.view(-1, self.num_classes)
        return score


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 5
    support_num_examples_per_class = 1
    query_num_examples_per_class = 4

    # Mimic Omniglot
    # model = RelationNetwork(1, 64, 128, 64, 64, num_classes=num_classes, support_num_examples_per_class=support_num_examples_per_class, query_num_examples_per_class=query_num_examples_per_class).to(device)
    # summary(model, input_size=[(num_classes, support_num_examples_per_class, 1, 28, 28), (num_classes, query_num_examples_per_class, 1, 28, 28)])

    # Mimic Mini Image Net Mimic
    filters_in = 1
    in_feat_rel = 64
    k = 20
    n = 1
    m = 19
    model = RelationNetwork(filters_in, 64, in_feat_rel, k, n, m)
    # model = RelationNetwork(3, 64, 128, 64, 576, num_classes=num_classes, support_num_examples_per_class=support_num_examples_per_class,
    #                         query_num_examples_per_class=query_num_examples_per_class).to(device)
    summary(model, input_size=[(k, n, 1, 28, 28), (k, m, 1, 28, 28)])
