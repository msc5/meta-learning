import torch
import torch.nn as nn

from torchinfo import summary


class Conv(nn.Module):

    def __init__(self, fi, fo, fs, skip, stride=None, padding=None):
        super(Conv, self).__init__()
        self.skip = skip
        conv = nn.Conv2d(fi, fo, fs, padding='same')
        if stride is not None:
            conv = nn.Conv2d(fi, fo, fs, stride=stride, padding=padding)
        self.stack = nn.Sequential(
            conv,
            nn.BatchNorm2d(fo),
            nn.ReLU()
        )

    def forward(self, x):
        res = x
        x = self.stack(x)
        if not self.skip:
            x = x + res
        return x


class ConvBlock(nn.Module):

    def __init__(self, fi, fo, fs, skip):
        super(ConvBlock, self).__init__()
        self.skip = skip
        self.stack = nn.Sequential(
            Conv(fi, fo, fs, skip),
            nn.ReLU(),
            Conv(fo, fo, fs, False),
        )

    def forward(self, x):
        res = x
        x = self.stack(x)
        if not self.skip:
            x = x + res
        return x


class ConvStack(nn.Module):

    def __init__(self, fi, fo, nb, skip):
        super(ConvStack, self).__init__()
        self.ent = ConvBlock(fi, fo, 3, skip)
        list = nn.ModuleList([ConvBlock(fo, fo, 3, False) for _ in range(nb)])
        list.append(nn.MaxPool2d(2, ceil_mode=True))
        self.stack = nn.Sequential(*list)

    def forward(self, x):
        x = self.ent(x)
        x = self.stack(x)
        return x


class ResNet(nn.Module):

    def __init__(self, fi, hi, wi, lo):
        super(ResNet, self).__init__()
        fo = 64
        self.stack = nn.Sequential(
            Conv(fi, fo, 7, True, stride=2, padding=3),
            nn.MaxPool2d(2, ceil_mode=True),
            ConvStack(fo, fo * 2**0, 3, True),
            ConvStack(fo * 2**0, fo * 2**1, 4, True),
            ConvStack(fo * 2**1, fo * 2**2, 6, True),
            ConvStack(fo * 2**2, fo * 2**3, 3, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.dec = nn.Linear(fo * 2**3, lo)

    def forward(self, x):
        x = self.stack(x)
        x = x.view(x.shape[0], -1)
        x = self.dec(x)
        return x


if __name__ == '__main__':
    hi = 105
    wi = 105
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet(3, hi, wi, 1000).to(device)
    actual = torch.hub.load('pytorch/vision:v0.10.0',
                            'resnet34', pretrained=True).to(device)
    summary(model, input_size=(3, hi, wi))
    summary(actual, input_size=(3, hi, wi))
