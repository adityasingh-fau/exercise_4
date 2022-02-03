from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.Relu = nn.ReLU()

        self.SkipConnection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, input):
        out = self.seq1(input)
        out = self.seq2(out)
        out += self.SkipConnection(input)
        out = self.Relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.Conv2D = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.BatchNorm = nn.BatchNorm2d(64)
        self.Relu = nn.ReLU()
        self.seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AvgPool2d(kernel_size=(10, 10)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.Conv2D(input)
        output = self.BatchNorm(output)
        output = self.Relu(output)
        output = self.seq(output)
        return output
