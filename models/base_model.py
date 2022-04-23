import sys
sys.path.append('..')
import torch.nn as nn
from utils.count_params import count_params

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, padding = 1, bias = True),
            nn.LeakyReLU(negative_slope = 0.125),
            nn.Conv2d(50, 50, 3, padding = 1, bias = True),
            nn.LeakyReLU(negative_slope = 0.125),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 50, 3, padding = 1, bias = True),
            nn.LeakyReLU(negative_slope = 0.125),
            nn.Conv2d(50, 50, 3, padding = 1, bias = True),
            nn.LeakyReLU(negative_slope = 0.125),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 50, 3, padding = 1, bias = True),
            nn.LeakyReLU(negative_slope = 0.125),
            nn.Conv2d(50, 1, 3, padding = 1, bias = True),
            nn.LeakyReLU(negative_slope = 0.125),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


if __name__ == '__main__':
    model = Predictor()
    count_params(model)
