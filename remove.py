from __future__ import print_function
import torch.utils.data
import math

from torch import nn

from Submodules.utils.utils_dcu import convbn, predict_normal, adaptative_cat

class UpProject(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size):
        super(UpProject, self).__init__()
        self.batch_size = batch_size

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), padding=(1, 1))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3), padding=(1, 1))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), padding=(1, 1))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        print(f"Input x shape: {x.shape}")

        out1_1 = self.conv1_1(x)
        out1_2 = self.conv1_2(x)
        out1_3 = self.conv1_3(x)
        out1_4 = self.conv1_4(x)

        out2_1 = self.conv2_1(x)
        out2_2 = self.conv2_2(x)
        out2_3 = self.conv2_3(x)
        out2_4 = self.conv2_4(x)

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            self.batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            self.batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            self.batch_size, -1, height * 2, width * 2)

        print(f"out1_1234 shape: {out1_1234.shape}")

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            self.batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            self.batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            self.batch_size, -1, height * 2, width * 2)

        print(f"out2_1234 shape: {out2_1234.shape}")

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        print(f"out1 after bn and conv3 shape: {out1.shape}")

        out2 = self.bn1_2(out2_1234)
        print(f"out2 after bn1_2 shape: {out2.shape}")

        out = out1 + out2
        out = self.relu(out)
        print(f"Final output shape: {out.shape}")

        return out