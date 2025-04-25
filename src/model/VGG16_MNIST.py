import torch.nn as nn

class VGG16_MNIST(nn.Module):
    def __init__(self):
        super(VGG16_MNIST, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.BatchNorm2d(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.BatchNorm2d(64)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer8 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer9 = nn.BatchNorm2d(128)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer12 = nn.BatchNorm2d(128)
        self.layer13 = nn.ReLU()
        self.layer14 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer15 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer16 = nn.BatchNorm2d(256)
        self.layer17 = nn.ReLU()
        self.layer18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer19 = nn.BatchNorm2d(256)
        self.layer20 = nn.ReLU()
        self.layer21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer22 = nn.BatchNorm2d(256)
        self.layer23 = nn.ReLU()
        self.layer24 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer25 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.layer26 = nn.BatchNorm2d(512)
        self.layer27 = nn.ReLU()
        self.layer28 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer29 = nn.BatchNorm2d(512)
        self.layer30 = nn.ReLU()
        self.layer31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer32 = nn.BatchNorm2d(512)
        self.layer33 = nn.ReLU()
        self.layer34 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer35 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer36 = nn.BatchNorm2d(512)
        self.layer37 = nn.ReLU()
        self.layer38 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer39 = nn.BatchNorm2d(512)
        self.layer40 = nn.ReLU()
        self.layer41 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer42 = nn.BatchNorm2d(512)
        self.layer43 = nn.ReLU()

        self.layer44 = nn.Flatten(1, -1)
        self.layer45 = nn.Dropout(0.5)
        self.layer46 = nn.Linear(512, 4096)  # input là 512x1x1
        self.layer47 = nn.ReLU()
        self.layer48 = nn.Dropout(0.5)
        self.layer49 = nn.Linear(4096, 4096)
        self.layer50 = nn.ReLU()
        self.layer51 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.layer6(self.layer5(self.layer4(x)))
        x = self.layer7(x)

        x = self.layer10(self.layer9(self.layer8(x)))
        x = self.layer13(self.layer12(self.layer11(x)))
        x = self.layer14(x)

        x = self.layer17(self.layer16(self.layer15(x)))
        x = self.layer20(self.layer19(self.layer18(x)))
        x = self.layer23(self.layer22(self.layer21(x)))
        x = self.layer24(x)

        x = self.layer27(self.layer26(self.layer25(x)))
        x = self.layer30(self.layer29(self.layer28(x)))
        x = self.layer33(self.layer32(self.layer31(x)))
        x = self.layer34(x)

        x = self.layer37(self.layer36(self.layer35(x)))
        x = self.layer40(self.layer39(self.layer38(x)))
        x = self.layer43(self.layer42(self.layer41(x)))

        x = self.layer44(x)
        x = self.layer45(x)
        x = self.layer46(x)
        x = self.layer47(x)
        x = self.layer48(x)
        x = self.layer49(x)
        x = self.layer50(x)
        x = self.layer51(x)
        return x