import torch
import torch.nn as nn

from torchinfo import summary


class SimpleConv(nn.Module):

    def __init__(self, in_size):
        super(SimpleConv, self).__init__()
        # 3x3 Convolution with 6 filters ('same' padding)
        self.conv = nn.Conv2d(in_size, 6, 3, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        # Applies convolution and then relu
        x = self.conv(x)
        x = self.relu(x)
        return x


class StackedConv(nn.Module):

    def __init__(self, in_size):
        super(StackedConv, self).__init__()
        # Stack 3 SimpleConv layers
        self.stack = nn.Sequential(
            SimpleConv(in_size),
            SimpleConv(6),
            SimpleConv(6),
        )

    def forward(self, x):
        x = self.stack(x)
        return x

# TODO: Abstract this away


class ResidualBlock(nn.Module):

    def __init__(self, in_size):
        super(ResidualBlock, self).__init__()
        self.conv = StackedConv(in_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = x + res
        return x


class StackedRes(nn.Module):

    def __init__(self, in_size):
        super(StackedRes, self).__init__()
        self.stack = nn.Sequential(
            StackedConv(in_size),
            ResidualBlock(6),
            ResidualBlock(6),
        )

    def forward(self, x):
        x = self.stack(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()
        self.flat = nn.Flatten()
        self.lin = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.flat(x)
        x = self.lin(x)
        return x


class ResNetwork(nn.Module):

    def __init__(self, in_size, in_dim, out_size):
        super(ResNetwork, self).__init__()
        self.__name__ = 'ResNetwork'
        self.res = StackedRes(in_size)
        self.enc = Encoder(6 * in_dim**2, out_size)

    def forward(self, x):
        x = self.res(x)
        x = self.enc(x)
        return x


if __name__ == '__main__':

    model = ResNetwork(1, 105, 1623)
    summary(model, input_size=(1, 105, 105))
