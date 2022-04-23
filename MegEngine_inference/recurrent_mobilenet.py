import numpy as np
import megengine.module as M
from megengine.utils.module_stats import module_stats

class MobileV2Layer(M.Module):
    def __init__(self, dim, expand_ratio=6):
        super(MobileV2Layer, self).__init__()

        hidden_dim = int(dim * expand_ratio)

        self.conv = M.Sequential(
            # pw
            M.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
            M.LeakyReLU(negative_slope=0.1),
            # dw
            M.Conv2d(hidden_dim, hidden_dim, 5, 1, 2, groups=hidden_dim, bias=False),
            M.LeakyReLU(negative_slope=0.1),
            # pw-linear
            M.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False))

    def forward(self, x):
        return x + self.conv(x)

# default dim=58, depth=2, unroll=48
class Predictor(M.Module):
    def __init__(self, dim=58, depth=2, unroll=48):
        super().__init__()
        self.unroll = unroll
        self.conv1 = M.Sequential(
            M.Conv2d(1, dim, 3, padding = 1, bias = False),
            M.LeakyReLU(negative_slope = 0.1),
        )
        layers = []
        for _ in range(depth):
            layers.append(MobileV2Layer(dim))
            layers.append(M.LeakyReLU(negative_slope=0.1))
        self.conv2 = M.Sequential(*layers)
        self.conv3 = M.Conv2d(dim, 1, 3, padding = 1, bias = False)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = out1
        for _ in range(self.unroll):
            out2 = self.conv2(out2) + out1
        out3 = self.conv3(out2)
        return out3

if __name__ == '__main__':
    model = Predictor(unroll=1)
    input_data = np.random.rand(1, 1, 256, 256).astype("float32")
    total_stats, stats_details = module_stats(
        model,
        inputs=(input_data,),
        cal_params=True,
        cal_flops=False,
        logging_to_stdout=True,
    )
