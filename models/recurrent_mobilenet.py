import sys
sys.path.append('..')
import torch.nn as nn
from utils.count_params import count_params

# kernel size of dconv : 3 -> 5
class MobileV2Layer(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(MobileV2Layer, self).__init__()

        hidden_dim = int(inp * expand_ratio)

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 5, 1, 2, groups=hidden_dim, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        )

    def forward(self, x):
            return x + self.conv(x)


# default dim=58, depth=2, unroll=48
class Predictor(nn.Module):
    def __init__(self, dim=58, depth=2, unroll=48):
        super().__init__()
        self.unroll = unroll
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, dim, 3, padding = 1, bias = False),
            nn.LeakyReLU(negative_slope = 0.1),
        )
        layers = []
        for _ in range(depth):
            layers.append(MobileV2Layer(dim, dim, expand_ratio=6))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        self.conv2 = nn.Sequential(*layers)

        self.conv3 = nn.Conv2d(dim, 1, 3, padding = 1, bias = False)
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = out1
        for _ in range(self.unroll):
            out2 = self.conv2(out2) + out1
        out3 = self.conv3(out2)
        return out3


if __name__ == '__main__':
    model = Predictor()
    count_params(model)