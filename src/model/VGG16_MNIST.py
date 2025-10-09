import torch.nn as nn

class VGG16_MNIST(nn.Module):
    def __init__(self, start_layer=0, end_layer=51):
        super(VGG16_MNIST, self).__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer

        if start_layer < 1 <= end_layer:
            self.layer1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        if start_layer < 2 <= end_layer:
            self.layer2 = nn.BatchNorm2d(64)
        if start_layer < 3 <= end_layer:
            self.layer3 = nn.ReLU()
        if start_layer < 4 <= end_layer:
            self.layer4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        if start_layer < 5 <= end_layer:
            self.layer5 = nn.BatchNorm2d(64)
        if start_layer < 6 <= end_layer:
            self.layer6 = nn.ReLU()
        if start_layer < 7 <= end_layer:
            self.layer7 = nn.MaxPool2d(kernel_size=2, stride=2)

        if start_layer < 8 <= end_layer:
            self.layer8 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        if start_layer < 9 <= end_layer:
            self.layer9 = nn.BatchNorm2d(128)
        if start_layer < 10 <= end_layer:
            self.layer10 = nn.ReLU()
        if start_layer < 11 <= end_layer:
            self.layer11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        if start_layer < 12 <= end_layer:
            self.layer12 = nn.BatchNorm2d(128)
        if start_layer < 13 <= end_layer:
            self.layer13 = nn.ReLU()
        if start_layer < 14 <= end_layer:
            self.layer14 = nn.MaxPool2d(kernel_size=2, stride=2)

        if start_layer < 15 <= end_layer:
            self.layer15 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        if start_layer < 16 <= end_layer:
            self.layer16 = nn.BatchNorm2d(256)
        if start_layer < 17 <= end_layer:
            self.layer17 = nn.ReLU()
        if start_layer < 18 <= end_layer:
            self.layer18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        if start_layer < 19 <= end_layer:
            self.layer19 = nn.BatchNorm2d(256)
        if start_layer < 20 <= end_layer:
            self.layer20 = nn.ReLU()
        if start_layer < 21 <= end_layer:
            self.layer21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        if start_layer < 22 <= end_layer:
            self.layer22 = nn.BatchNorm2d(256)
        if start_layer < 23 <= end_layer:
            self.layer23 = nn.ReLU()
        if start_layer < 24 <= end_layer:
            self.layer24 = nn.MaxPool2d(kernel_size=2, stride=2)

        if start_layer < 25 <= end_layer:
            self.layer25 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 26 <= end_layer:
            self.layer26 = nn.BatchNorm2d(512)
        if start_layer < 27 <= end_layer:
            self.layer27 = nn.ReLU()
        if start_layer < 28 <= end_layer:
            self.layer28 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 29 <= end_layer:
            self.layer29 = nn.BatchNorm2d(512)
        if start_layer < 30 <= end_layer:
            self.layer30 = nn.ReLU()
        if start_layer < 31 <= end_layer:
            self.layer31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 32 <= end_layer:
            self.layer32 = nn.BatchNorm2d(512)
        if start_layer < 33 <= end_layer:
            self.layer33 = nn.ReLU()
        if start_layer < 34 <= end_layer:
            self.layer34 = nn.MaxPool2d(kernel_size=2, stride=2)

        if start_layer < 35 <= end_layer:
            self.layer35 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 36 <= end_layer:
            self.layer36 = nn.BatchNorm2d(512)
        if start_layer < 37 <= end_layer:
            self.layer37 = nn.ReLU()
        if start_layer < 38 <= end_layer:
            self.layer38 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 39 <= end_layer:
            self.layer39 = nn.BatchNorm2d(512)
        if start_layer < 40 <= end_layer:
            self.layer40 = nn.ReLU()
        if start_layer < 41 <= end_layer:
            self.layer41 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if start_layer < 42 <= end_layer:
            self.layer42 = nn.BatchNorm2d(512)
        if start_layer < 43 <= end_layer:
            self.layer43 = nn.ReLU()

        if start_layer < 44 <= end_layer:
            self.layer44 = nn.Flatten(1, -1)
        if start_layer < 45 <= end_layer:
            self.layer45 = nn.Dropout(0.5)
        if start_layer < 46 <= end_layer:
            self.layer46 = nn.Linear(512, 4096)
        if start_layer < 47 <= end_layer:
            self.layer47 = nn.ReLU()
        if start_layer < 48 <= end_layer:
            self.layer48 = nn.Dropout(0.5)
        if start_layer < 49 <= end_layer:
            self.layer49 = nn.Linear(4096, 4096)
        if start_layer < 50 <= end_layer:
            self.layer50 = nn.ReLU()
        if start_layer < 51 <= end_layer:
            self.layer51 = nn.Linear(4096, 10)

    def forward(self, x):
        if self.start_layer < 1 <= self.end_layer:
            x = self.layer1(x)
        if self.start_layer < 2 <= self.end_layer:
            x = self.layer2(x)
        if self.start_layer < 3 <= self.end_layer:
            x = self.layer3(x)
        if self.start_layer < 4 <= self.end_layer:
            x = self.layer4(x)
        if self.start_layer < 5 <= self.end_layer:
            x = self.layer5(x)
        if self.start_layer < 6 <= self.end_layer:
            x = self.layer6(x)
        if self.start_layer < 7 <= self.end_layer:
            x = self.layer7(x)

        if self.start_layer < 8 <= self.end_layer:
            x = self.layer8(x)
        if self.start_layer < 9 <= self.end_layer:
            x = self.layer9(x)
        if self.start_layer < 10 <= self.end_layer:
            x = self.layer10(x)
        if self.start_layer < 11 <= self.end_layer:
            x = self.layer11(x)
        if self.start_layer < 12 <= self.end_layer:
            x = self.layer12(x)
        if self.start_layer < 13 <= self.end_layer:
            x = self.layer13(x)
        if self.start_layer < 14 <= self.end_layer:
            x = self.layer14(x)

        if self.start_layer < 15 <= self.end_layer:
            x = self.layer15(x)
        if self.start_layer < 16 <= self.end_layer:
            x = self.layer16(x)
        if self.start_layer < 17 <= self.end_layer:
            x = self.layer17(x)
        if self.start_layer < 18 <= self.end_layer:
            x = self.layer18(x)
        if self.start_layer < 19 <= self.end_layer:
            x = self.layer19(x)
        if self.start_layer < 20 <= self.end_layer:
            x = self.layer20(x)
        if self.start_layer < 21 <= self.end_layer:
            x = self.layer21(x)
        if self.start_layer < 22 <= self.end_layer:
            x = self.layer22(x)
        if self.start_layer < 23 <= self.end_layer:
            x = self.layer23(x)
        if self.start_layer < 24 <= self.end_layer:
            x = self.layer24(x)

        if self.start_layer < 25 <= self.end_layer:
            x = self.layer25(x)
        if self.start_layer < 26 <= self.end_layer:
            x = self.layer26(x)
        if self.start_layer < 27 <= self.end_layer:
            x = self.layer27(x)
        if self.start_layer < 28 <= self.end_layer:
            x = self.layer28(x)
        if self.start_layer < 29 <= self.end_layer:
            x = self.layer29(x)
        if self.start_layer < 30 <= self.end_layer:
            x = self.layer30(x)
        if self.start_layer < 31 <= self.end_layer:
            x = self.layer31(x)
        if self.start_layer < 32 <= self.end_layer:
            x = self.layer32(x)
        if self.start_layer < 33 <= self.end_layer:
            x = self.layer33(x)
        if self.start_layer < 34 <= self.end_layer:
            x = self.layer34(x)

        if self.start_layer < 35 <= self.end_layer:
            x = self.layer35(x)
        if self.start_layer < 36 <= self.end_layer:
            x = self.layer36(x)
        if self.start_layer < 37 <= self.end_layer:
            x = self.layer37(x)
        if self.start_layer < 38 <= self.end_layer:
            x = self.layer38(x)
        if self.start_layer < 39 <= self.end_layer:
            x = self.layer39(x)
        if self.start_layer < 40 <= self.end_layer:
            x = self.layer40(x)
        if self.start_layer < 41 <= self.end_layer:
            x = self.layer41(x)
        if self.start_layer < 42 <= self.end_layer:
            x = self.layer42(x)
        if self.start_layer < 43 <= self.end_layer:
            x = self.layer43(x)

        if self.start_layer < 44 <= self.end_layer:
            x = self.layer44(x)
        if self.start_layer < 45 <= self.end_layer:
            x = self.layer45(x)
        if self.start_layer < 46 <= self.end_layer:
            x = self.layer46(x)
        if self.start_layer < 47 <= self.end_layer:
            x = self.layer47(x)
        if self.start_layer < 48 <= self.end_layer:
            x = self.layer48(x)
        if self.start_layer < 49 <= self.end_layer:
            x = self.layer49(x)
        if self.start_layer < 50 <= self.end_layer:
            x = self.layer50(x)
        if self.start_layer < 51 <= self.end_layer:
            x = self.layer51(x)

        return x
