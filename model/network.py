import torch
import torch.nn as nn
from model.modules.dense_block import DenseBlock

class CNN_Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),
            nn.Dropout1d(0.1)
        )
    def forward(self, x):
        return self.block(x)

class BaseLine_CNN_follow_JBHI2024(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.conv1 = CNN_Block(in_channels,16,5)
        self.conv2 = CNN_Block(16,32,3)
        self.conv3 = CNN_Block(32,32,3)
        self.conv4 = CNN_Block(32,256,3)
        self.conv5 = CNN_Block(256,512,3)
        self.conv6 = CNN_Block(512,512,3)
        self.avg_pool = nn.AdaptiveMaxPool1d(1)

        self.dense4 = DenseBlock(num_layers=6,num_input_features=512,bn_size=4,growth_rate=32,drop_rate=0)
        self.classifier = nn.Conv1d(704,num_classes,1)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x_gmp = self.avg_pool(x6)
        x_dense4 = self.dense4(x_gmp)
        out = self.classifier(x_dense4)
        return out

if __name__ == '__main__':
    input = torch.randn(4,1,5000)
    net = BaseLine_CNN_follow_JBHI2024(in_channels=1,num_classes=2)
    outputs = net(input)
    outputs = torch.sigmoid(outputs)
    preds = (outputs[:, 1, :] > 0.5).to(torch.int8).squeeze()
    print(preds.shape)