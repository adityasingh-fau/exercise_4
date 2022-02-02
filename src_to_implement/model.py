from torch import nn


class ResNet(nn.Module):
    def __init__(self):
        super(self, ResNet).__init__()
        self.Conv2D = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.BatchNorm = nn.BatchNorm2d()
        self.Relu = nn.ReLU()
        self.seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AvgPool2d(kernel_size=(10, 10)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.Conv2D(input)
        output = self.BatchNorm(output)
        output = self.Relu(output)
        output = self.seq(output)
        return output


class ResBlock:
    def __init__(self, in_channels, out_channels, stride):
        super(self, ResBlock).__init__()

        self.seq1 = nn.sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
            nn.BatchNorm2d(),
            nn.ReLU()
        )
        self.seq2 = nn.sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(),
        )

        self.SkipConnection = nn.sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(),
        )

    def forward(self, input):
        out = self.seq1(input)
        out = self.seq2(out)
        out += self.SkipConnection(input)
        out = nn.ReLU(out)
        return out
