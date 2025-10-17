import torch
import torch.nn as nn

class MobileNetv1_MNIST(nn.Module):
    def __init__(self, start_layer=0, end_layer=84):
        super(MobileNetv1_MNIST, self).__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer

        if start_layer < 1 <= end_layer:
            self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        if start_layer < 2 <= end_layer:
            self.layer2 = nn.BatchNorm2d(32)
        if start_layer < 3 <= end_layer:
            self.layer3 = nn.ReLU()
        if start_layer < 4 <= end_layer:
            self.layer4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        if start_layer < 5 <= end_layer:
            self.layer5 = nn.BatchNorm2d(32)
        if start_layer < 6 <= end_layer:
            self.layer6 = nn.ReLU()
        if start_layer < 7 <= end_layer:
            self.layer7 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        if start_layer < 8 <= end_layer:
            self.layer8 = nn.BatchNorm2d(64)
        if start_layer < 9 <= end_layer:
            self.layer9 = nn.ReLU()
        if start_layer < 10 <= end_layer:
            self.layer10 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        if start_layer < 11 <= end_layer:
            self.layer11 = nn.BatchNorm2d(64)
        if start_layer < 12 <= end_layer:
            self.layer12 = nn.ReLU()
        if start_layer < 13 <= end_layer:
            self.layer13 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        if start_layer < 14 <= end_layer:
            self.layer14 = nn.BatchNorm2d(128)
        if start_layer < 15 <= end_layer:
            self.layer15 = nn.ReLU()
        if start_layer < 16 <= end_layer:
            self.layer16 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        if start_layer < 17 <= end_layer:
            self.layer17 = nn.BatchNorm2d(128)
        if start_layer < 18 <= end_layer:
            self.layer18 = nn.ReLU()
        if start_layer < 19 <= end_layer:
            self.layer19 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        if start_layer < 20 <= end_layer:
            self.layer20 = nn.BatchNorm2d(128)
        if start_layer < 21 <= end_layer:
            self.layer21 = nn.ReLU()
        if start_layer < 22 <= end_layer:
            self.layer22 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        if start_layer < 23 <= end_layer:
            self.layer23 = nn.BatchNorm2d(128)
        if start_layer < 24 <= end_layer:
            self.layer24 = nn.ReLU()
        if start_layer < 25 <= end_layer:
            self.layer25 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        if start_layer < 26 <= end_layer:
            self.layer26 = nn.BatchNorm2d(256)
        if start_layer < 27 <= end_layer:
            self.layer27 = nn.ReLU()
        if start_layer < 28 <= end_layer:
            self.layer28 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        if start_layer < 29 <= end_layer:
            self.layer29 = nn.BatchNorm2d(256)
        if start_layer < 30 <= end_layer:
            self.layer30 = nn.ReLU()
        if start_layer < 31 <= end_layer:
            self.layer31 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        if start_layer < 32 <= end_layer:
            self.layer32 = nn.BatchNorm2d(256)
        if start_layer < 33 <= end_layer:
            self.layer33 = nn.ReLU()
        if start_layer < 34 <= end_layer:
            self.layer34 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        if start_layer < 35 <= end_layer:
            self.layer35 = nn.BatchNorm2d(256)
        if start_layer < 36 <= end_layer:
            self.layer36 = nn.ReLU()
        if start_layer < 37 <= end_layer:
            self.layer37 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
        if start_layer < 38 <= end_layer:
            self.layer38 = nn.BatchNorm2d(512)
        if start_layer < 39 <= end_layer:
            self.layer39 = nn.ReLU()
        if start_layer < 40 <= end_layer:
            self.layer40 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 41 <= end_layer:
            self.layer41 = nn.BatchNorm2d(512)
        if start_layer < 42 <= end_layer:
            self.layer42 = nn.ReLU()
        if start_layer < 43 <= end_layer:
            self.layer43 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        if start_layer < 44 <= end_layer:
            self.layer44 = nn.BatchNorm2d(512)
        if start_layer < 45 <= end_layer:
            self.layer45 = nn.ReLU()
        if start_layer < 46 <= end_layer:
            self.layer46 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 47 <= end_layer:
            self.layer47 = nn.BatchNorm2d(512)
        if start_layer < 48 <= end_layer:
            self.layer48 = nn.ReLU()
        if start_layer < 49 <= end_layer:
            self.layer49 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        if start_layer < 50 <= end_layer:
            self.layer50 = nn.BatchNorm2d(512)
        if start_layer < 51 <= end_layer:
            self.layer51 = nn.ReLU()
        if start_layer < 52 <= end_layer:
            self.layer52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 53 <= end_layer:
            self.layer53 = nn.BatchNorm2d(512)
        if start_layer < 54 <= end_layer:
            self.layer54 = nn.ReLU()
        if start_layer < 55 <= end_layer:
            self.layer55 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        if start_layer < 56 <= end_layer:
            self.layer56 = nn.BatchNorm2d(512)
        if start_layer < 57 <= end_layer:
            self.layer57 = nn.ReLU()
        if start_layer < 58 <= end_layer:
            self.layer58 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 59 <= end_layer:
            self.layer59 = nn.BatchNorm2d(512)
        if start_layer < 60 <= end_layer:
            self.layer60 = nn.ReLU()
        if start_layer < 61 <= end_layer:
            self.layer61 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        if start_layer < 62 <= end_layer:
            self.layer62 = nn.BatchNorm2d(512)
        if start_layer < 63 <= end_layer:
            self.layer63 = nn.ReLU()
        if start_layer < 64 <= end_layer:
            self.layer64 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 65 <= end_layer:
            self.layer65 = nn.BatchNorm2d(512)
        if start_layer < 66 <= end_layer:
            self.layer66 = nn.ReLU()
        if start_layer < 67 <= end_layer:
            self.layer67 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        if start_layer < 68 <= end_layer:
            self.layer68 = nn.BatchNorm2d(512)
        if start_layer < 69 <= end_layer:
            self.layer69 = nn.ReLU()
        if start_layer < 70 <= end_layer:
            self.layer70 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        if start_layer < 71 <= end_layer:
            self.layer71 = nn.BatchNorm2d(512)
        if start_layer < 72 <= end_layer:
            self.layer72 = nn.ReLU()
        if start_layer < 73 <= end_layer:
            self.layer73 = nn.Conv2d(512, 1024, kernel_size=1, stride=1)
        if start_layer < 74 <= end_layer:
            self.layer74 = nn.BatchNorm2d(1024)
        if start_layer < 75 <= end_layer:
            self.layer75 = nn.ReLU()
        if start_layer < 76 <= end_layer:
            self.layer76 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        if start_layer < 77 <= end_layer:
            self.layer77 = nn.BatchNorm2d(1024)
        if start_layer < 78 <= end_layer:
            self.layer78 = nn.ReLU()
        if start_layer < 79 <= end_layer:
            self.layer79 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        if start_layer < 80 <= end_layer:
            self.layer80 = nn.BatchNorm2d(1024)
        if start_layer < 81 <= end_layer:
            self.layer81 = nn.ReLU()
        if start_layer < 82 <= end_layer:
            self.layer82 = nn.MaxPool2d(2, 2)
        if start_layer < 83 <= end_layer:
            self.layer83 = nn.Flatten(1, -1)
        if start_layer < 84 <= end_layer:
            self.layer84 = nn.Linear(1024, 10)

    def forward(self, x):
        for i in range(1, 85):
            if self.start_layer < i <= self.end_layer:
                layer = getattr(self, f'layer{i}', None)
                if layer is not None:
                    x = layer(x)
        return x
