import sys
sys.path.append('..')
import torch.nn as nn
from utils.count_params import count_params

# default dim=50, depth=4, unroll=12
class Predictor(nn.Module):
    def __init__(self, dim=50, depth=4, unroll=6):
        super().__init__()
        self.unroll = unroll
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, dim, 3, padding = 1, bias = True),
            nn.LeakyReLU(negative_slope = 0.125),
        )

        layers = []
        for _ in range(depth):
            layers.append(nn.Conv2d(dim, dim, 3, padding=1, bias=True))
            layers.append(nn.LeakyReLU(negative_slope=0.125))
        self.conv2 = nn.Sequential(*layers)

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, 1, 3, padding = 1, bias = True),
        )
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
