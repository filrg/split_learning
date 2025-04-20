import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print(f"Using device: CPU")


class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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
        self.layer44 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer45 = nn.Flatten(1, -1)
        self.layer46 = nn.Dropout(0.5)
        self.layer47 = nn.Linear(1 * 1 * 512, 4096)
        self.layer48 = nn.ReLU()
        self.layer49 = nn.Dropout(0.5)
        self.layer50 = nn.Linear(4096, 4096)
        self.layer51 = nn.ReLU()
        self.layer52 = nn.Linear(4096, 10)

    def forward(self, input0):
        out1 = self.layer1(input0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out44 = self.layer44(out43)
        out45 = self.layer45(out44)
        out46 = self.layer46(out45)
        out47 = self.layer47(out46)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out51 = self.layer51(out50)
        out52 = self.layer52(out51)
        return out52

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

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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
        self.layer24 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer25 = nn.BatchNorm2d(256)
        self.layer26 = nn.ReLU()
        self.layer27 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer28 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.layer29 = nn.BatchNorm2d(512)
        self.layer30 = nn.ReLU()
        self.layer31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer32 = nn.BatchNorm2d(512)
        self.layer33 = nn.ReLU()
        self.layer34 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer35 = nn.BatchNorm2d(512)
        self.layer36 = nn.ReLU()
        self.layer37 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer38 = nn.BatchNorm2d(512)
        self.layer39 = nn.ReLU()
        self.layer40 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer41 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer42 = nn.BatchNorm2d(512)
        self.layer43 = nn.ReLU()
        self.layer44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer45 = nn.BatchNorm2d(512)
        self.layer46 = nn.ReLU()
        self.layer47 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer48 = nn.BatchNorm2d(512)
        self.layer49 = nn.ReLU()
        self.layer50 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer51 = nn.BatchNorm2d(512)
        self.layer52 = nn.ReLU()
        self.layer53 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer54 = nn.Flatten(1, -1)
        self.layer55 = nn.Dropout(0.5)
        self.layer56 = nn.Linear(1 * 1 * 512, 4096)
        self.layer57 = nn.ReLU()
        self.layer58 = nn.Dropout(0.5)
        self.layer59 = nn.Linear(4096, 4096)
        self.layer60 = nn.ReLU()
        self.layer61 = nn.Linear(4096, 10)

    def forward(self, input0):
        out1 = self.layer1(input0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out44 = self.layer44(out43)
        out45 = self.layer45(out44)
        out46 = self.layer46(out45)
        out47 = self.layer47(out46)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out51 = self.layer51(out50)
        out52 = self.layer52(out51)
        out53 = self.layer53(out52)
        out54 = self.layer54(out53)
        out55 = self.layer55(out54)
        out56 = self.layer56(out55)
        out57 = self.layer57(out56)
        out58 = self.layer58(out57)
        out59 = self.layer59(out58)
        out60 = self.layer60(out59)
        out61 = self.layer61(out60)
        return out61


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 6, 5)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.MaxPool2d(2, 2)
        self.layer4 = nn.Conv2d(6, 16, 5)
        self.layer5 = nn.ReLU()
        self.layer6 = nn.MaxPool2d(2, 2)
        self.layer7 = nn.Flatten(1, -1)
        self.layer8 = nn.Linear(16 * 5 * 5, 120)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.Linear(120, 84)
        self.layer11 = nn.ReLU()
        self.layer12 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        return out


class MobileNetv1(torch.nn.Module):
    def __init__(self):
        super(MobileNetv1, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.BatchNorm2d(32)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.BatchNorm2d(32)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.layer8 = nn.BatchNorm2d(64)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer11 = nn.BatchNorm2d(64)
        self.layer12 = nn.ReLU()
        self.layer13 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.layer14 = nn.BatchNorm2d(128)
        self.layer15 = nn.ReLU()
        self.layer16 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer17 = nn.BatchNorm2d(128)
        self.layer18 = nn.ReLU()
        self.layer19 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.layer20 = nn.BatchNorm2d(128)
        self.layer21 = nn.ReLU()
        self.layer22 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.layer23 = nn.BatchNorm2d(128)
        self.layer24 = nn.ReLU()
        self.layer25 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.layer26 = nn.BatchNorm2d(256)
        self.layer27 = nn.ReLU()
        self.layer28 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer29 = nn.BatchNorm2d(256)
        self.layer30 = nn.ReLU()
        self.layer31 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.layer32 = nn.BatchNorm2d(256)
        self.layer33 = nn.ReLU()
        self.layer34 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.layer35 = nn.BatchNorm2d(256)
        self.layer36 = nn.ReLU()
        self.layer37 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
        self.layer38 = nn.BatchNorm2d(512)
        self.layer39 = nn.ReLU()
        self.layer40 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer41 = nn.BatchNorm2d(512)
        self.layer42 = nn.ReLU()
        self.layer43 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.layer44 = nn.BatchNorm2d(512)
        self.layer45 = nn.ReLU()
        self.layer46 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer47 = nn.BatchNorm2d(512)
        self.layer48 = nn.ReLU()
        self.layer49 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.layer50 = nn.BatchNorm2d(512)
        self.layer51 = nn.ReLU()
        self.layer52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer53 = nn.BatchNorm2d(512)
        self.layer54 = nn.ReLU()
        self.layer55 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.layer56 = nn.BatchNorm2d(512)
        self.layer57 = nn.ReLU()
        self.layer58 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer59 = nn.BatchNorm2d(512)
        self.layer60 = nn.ReLU()
        self.layer61 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.layer62 = nn.BatchNorm2d(512)
        self.layer63 = nn.ReLU()
        self.layer64 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer65 = nn.BatchNorm2d(512)
        self.layer66 = nn.ReLU()
        self.layer67 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.layer68 = nn.BatchNorm2d(512)
        self.layer69 = nn.ReLU()
        self.layer70 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.layer71 = nn.BatchNorm2d(512)
        self.layer72 = nn.ReLU()
        self.layer73 = nn.Conv2d(512, 1024, kernel_size=1, stride=1)
        self.layer74 = nn.BatchNorm2d(1024)
        self.layer75 = nn.ReLU()
        self.layer76 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.layer77 = nn.BatchNorm2d(1024)
        self.layer78 = nn.ReLU()
        self.layer79 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.layer80 = nn.BatchNorm2d(1024)
        self.layer81 = nn.ReLU()
        self.layer82 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer83 = nn.Flatten(1, -1)
        self.layer84 = nn.Linear(1024, 10)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out44 = self.layer44(out43)
        out45 = self.layer45(out44)
        out46 = self.layer46(out45)
        out47 = self.layer47(out46)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out51 = self.layer51(out50)
        out52 = self.layer52(out51)
        out53 = self.layer53(out52)
        out54 = self.layer54(out53)
        out55 = self.layer55(out54)
        out56 = self.layer56(out55)
        out57 = self.layer57(out56)
        out58 = self.layer58(out57)
        out59 = self.layer59(out58)
        out60 = self.layer60(out59)
        out61 = self.layer61(out60)
        out62 = self.layer62(out61)
        out63 = self.layer63(out62)
        out64 = self.layer64(out63)
        out65 = self.layer65(out64)
        out66 = self.layer66(out65)
        out67 = self.layer67(out66)
        out68 = self.layer68(out67)
        out69 = self.layer69(out68)
        out70 = self.layer70(out69)
        out71 = self.layer71(out70)
        out72 = self.layer72(out71)
        out73 = self.layer73(out72)
        out74 = self.layer74(out73)
        out75 = self.layer75(out74)
        out76 = self.layer76(out75)
        out77 = self.layer77(out76)
        out78 = self.layer78(out77)
        out79 = self.layer79(out78)
        out80 = self.layer80(out79)
        out81 = self.layer81(out80)
        out82 = self.layer82(out81)
        out83 = self.layer83(out82)
        out84 = self.layer84(out83)
        return out84

class MobileNetV1_MNIST(nn.Module):
    def __init__(self):
        super(MobileNetV1_MNIST, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.BatchNorm2d(32)
        self.layer3 = nn.ReLU()

        self.layer4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.BatchNorm2d(64)
        self.layer6 = nn.ReLU()

        self.layer7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 28 → 14
        self.layer8 = nn.BatchNorm2d(128)
        self.layer9 = nn.ReLU()

        self.layer10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer11 = nn.BatchNorm2d(128)
        self.layer12 = nn.ReLU()

        self.layer13 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 14 → 7
        self.layer14 = nn.BatchNorm2d(256)
        self.layer15 = nn.ReLU()

        self.layer16 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer17 = nn.BatchNorm2d(256)
        self.layer18 = nn.ReLU()

        self.layer19 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 7 → 4
        self.layer20 = nn.BatchNorm2d(512)
        self.layer21 = nn.ReLU()

        self.layer22 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer23 = nn.BatchNorm2d(512)
        self.layer24 = nn.ReLU()

        self.layer25 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer26 = nn.BatchNorm2d(512)
        self.layer27 = nn.ReLU()

        self.layer28 = nn.AvgPool2d(kernel_size=4)  # 4x4 → 1x1

        self.layer29 = nn.Flatten(1, -1)
        self.layer30 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))       # 28x28
        x = self.layer6(self.layer5(self.layer4(x)))       # 28x28
        x = self.layer9(self.layer8(self.layer7(x)))       # 14x14
        x = self.layer12(self.layer11(self.layer10(x)))    # 14x14
        x = self.layer15(self.layer14(self.layer13(x)))    # 7x7
        x = self.layer18(self.layer17(self.layer16(x)))    # 7x7
        x = self.layer21(self.layer20(self.layer19(x)))    # 4x4
        x = self.layer24(self.layer23(self.layer22(x)))    # 4x4
        x = self.layer27(self.layer26(self.layer25(x)))    # 4x4
        x = self.layer28(x)                                # 1x1
        x = self.layer29(x)
        x = self.layer30(x)
        return x


class ViT_CIFAR10_Deep6(nn.Module):
    def __init__(self):
        super(ViT_CIFAR10_Deep6, self).__init__()
        img_size = 32
        patch_size = 4
        in_channels = 3
        embed_dim = 128
        num_classes = 10
        num_patches = (img_size // patch_size) ** 2

        # ----- Patch Embedding -----
        self.layer1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # [B, 128, 8, 8]
        self.layer2 = nn.Flatten(2)  # [B, 128, 64]
        self.layer3 = nn.Identity()  # transpose in forward

        # ----- CLS token + Positional embedding -----
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.layer4 = nn.Identity()  # cls concat
        self.layer5 = nn.Identity()  # pos_embed addition

        # ---------- Encoder Block 1 ----------
        self.layer6 = nn.LayerNorm(embed_dim)
        self.layer7 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer8 = nn.Identity()
        self.layer9 = nn.LayerNorm(embed_dim)
        self.layer10 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer11 = nn.Identity()

        # ---------- Encoder Block 2 ----------
        self.layer12 = nn.LayerNorm(embed_dim)
        self.layer13 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer14 = nn.Identity()
        self.layer15 = nn.LayerNorm(embed_dim)
        self.layer16 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer17 = nn.Identity()

        # ---------- Encoder Block 3 ----------
        self.layer18 = nn.LayerNorm(embed_dim)
        self.layer19 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer20 = nn.Identity()
        self.layer21 = nn.LayerNorm(embed_dim)
        self.layer22 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer23 = nn.Identity()

        # ---------- Encoder Block 4 ----------
        self.layer24 = nn.LayerNorm(embed_dim)
        self.layer25 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer26 = nn.Identity()
        self.layer27 = nn.LayerNorm(embed_dim)
        self.layer28 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer29 = nn.Identity()

        # ---------- Encoder Block 5 ----------
        self.layer30 = nn.LayerNorm(embed_dim)
        self.layer31 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer32 = nn.Identity()
        self.layer33 = nn.LayerNorm(embed_dim)
        self.layer34 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer35 = nn.Identity()

        # ---------- Encoder Block 6 ----------
        self.layer36 = nn.LayerNorm(embed_dim)
        self.layer37 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer38 = nn.Identity()
        self.layer39 = nn.LayerNorm(embed_dim)
        self.layer40 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer41 = nn.Identity()

        # ---------- Classification Head ----------
        self.layer42 = nn.LayerNorm(embed_dim)
        self.layer43 = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = out2.transpose(1, 2)

        cls_token = self.cls_token.expand(B, -1, -1)
        out4 = self.layer4(torch.cat([cls_token, out3], dim=1))
        out5 = self.layer5(out4 + self.pos_embed)

        # Block 1
        x = self.layer6(out5)
        x_attn, _ = self.layer7(x, x, x)
        x = self.layer8(x_attn + out5)
        x_mlp = self.layer10(self.layer9(x))
        x = self.layer11(x + x_mlp)

        # Block 2
        x_ = self.layer12(x)
        x_attn, _ = self.layer13(x_, x_, x_)
        x = self.layer14(x_attn + x)
        x_mlp = self.layer16(self.layer15(x))
        x = self.layer17(x + x_mlp)

        # Block 3
        x_ = self.layer18(x)
        x_attn, _ = self.layer19(x_, x_, x_)
        x = self.layer20(x_attn + x)
        x_mlp = self.layer22(self.layer21(x))
        x = self.layer23(x + x_mlp)

        # Block 4
        x_ = self.layer24(x)
        x_attn, _ = self.layer25(x_, x_, x_)
        x = self.layer26(x_attn + x)
        x_mlp = self.layer28(self.layer27(x))
        x = self.layer29(x + x_mlp)

        # Block 5
        x_ = self.layer30(x)
        x_attn, _ = self.layer31(x_, x_, x_)
        x = self.layer32(x_attn + x)
        x_mlp = self.layer34(self.layer33(x))
        x = self.layer35(x + x_mlp)

        # Block 6
        x_ = self.layer36(x)
        x_attn, _ = self.layer37(x_, x_, x_)
        x = self.layer38(x_attn + x)
        x_mlp = self.layer40(self.layer39(x))
        x = self.layer41(x + x_mlp)

        out_cls = self.layer42(x[:, 0])
        out = self.layer43(out_cls)
        return out

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        img_size = 32
        patch_size = 4
        in_channels = 3
        embed_dim = 128
        num_classes = 10
        num_patches = (img_size // patch_size) ** 2

        # ----- Patch Embedding -----
        self.layer1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # [B, 128, 8, 8]
        self.layer2 = nn.Flatten(2)  # [B, 128, 64]
        self.layer3 = nn.Identity()  # transpose in forward

        # ----- CLS token + Positional embedding -----
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.layer4 = nn.Identity()  # cls concat
        self.layer5 = nn.Identity()  # pos_embed addition

        # ---------- Encoder Block 1 ----------
        self.layer6 = nn.LayerNorm(embed_dim)
        self.layer7 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer8 = nn.Identity()
        self.layer9 = nn.LayerNorm(embed_dim)
        self.layer10 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer11 = nn.Identity()

        # ---------- Encoder Block 2 ----------
        self.layer12 = nn.LayerNorm(embed_dim)
        self.layer13 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer14 = nn.Identity()
        self.layer15 = nn.LayerNorm(embed_dim)
        self.layer16 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer17 = nn.Identity()

        # ---------- Encoder Block 3 ----------
        self.layer18 = nn.LayerNorm(embed_dim)
        self.layer19 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer20 = nn.Identity()
        self.layer21 = nn.LayerNorm(embed_dim)
        self.layer22 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer23 = nn.Identity()

        # ---------- Encoder Block 4 ----------
        self.layer24 = nn.LayerNorm(embed_dim)
        self.layer25 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer26 = nn.Identity()
        self.layer27 = nn.LayerNorm(embed_dim)
        self.layer28 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer29 = nn.Identity()

        # ---------- Encoder Block 5 ----------
        self.layer30 = nn.LayerNorm(embed_dim)
        self.layer31 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer32 = nn.Identity()
        self.layer33 = nn.LayerNorm(embed_dim)
        self.layer34 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer35 = nn.Identity()

        # ---------- Encoder Block 6 ----------
        self.layer36 = nn.LayerNorm(embed_dim)
        self.layer37 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer38 = nn.Identity()
        self.layer39 = nn.LayerNorm(embed_dim)
        self.layer40 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer41 = nn.Identity()

        # ---------- Classification Head ----------
        self.layer42 = nn.LayerNorm(embed_dim)
        self.layer43 = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = out2.transpose(1, 2)

        cls_token = self.cls_token.expand(B, -1, -1)
        out4 = self.layer4(torch.cat([cls_token, out3], dim=1))
        out5 = self.layer5(out4 + self.pos_embed)

        # Block 1
        x = self.layer6(out5)
        x_attn, _ = self.layer7(x, x, x)
        x = self.layer8(x_attn + out5)
        x_mlp = self.layer10(self.layer9(x))
        x = self.layer11(x + x_mlp)

        # Block 2
        x_ = self.layer12(x)
        x_attn, _ = self.layer13(x_, x_, x_)
        x = self.layer14(x_attn + x)
        x_mlp = self.layer16(self.layer15(x))
        x = self.layer17(x + x_mlp)

        # Block 3
        x_ = self.layer18(x)
        x_attn, _ = self.layer19(x_, x_, x_)
        x = self.layer20(x_attn + x)
        x_mlp = self.layer22(self.layer21(x))
        x = self.layer23(x + x_mlp)

        # Block 4
        x_ = self.layer24(x)
        x_attn, _ = self.layer25(x_, x_, x_)
        x = self.layer26(x_attn + x)
        x_mlp = self.layer28(self.layer27(x))
        x = self.layer29(x + x_mlp)

        # Block 5
        x_ = self.layer30(x)
        x_attn, _ = self.layer31(x_, x_, x_)
        x = self.layer32(x_attn + x)
        x_mlp = self.layer34(self.layer33(x))
        x = self.layer35(x + x_mlp)

        # Block 6
        x_ = self.layer36(x)
        x_attn, _ = self.layer37(x_, x_, x_)
        x = self.layer38(x_attn + x)
        x_mlp = self.layer40(self.layer39(x))
        x = self.layer41(x + x_mlp)

        out_cls = self.layer42(x[:, 0])
        out = self.layer43(out_cls)
        return out

class ViT_MNIST(nn.Module):
    def __init__(self):
        super(ViT_MNIST, self).__init__()
        img_size = 28
        patch_size = 4
        in_channels = 1
        embed_dim = 128
        num_classes = 10
        num_patches = (img_size // patch_size) ** 2

        # ----- Patch Embedding -----
        self.layer1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # [B, 128, 7, 7]
        self.layer2 = nn.Flatten(2)  # [B, 128, 49]
        self.layer3 = nn.Identity()  # sẽ dùng transpose trong forward()

        # ----- CLS token + Positional embedding -----
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.layer4 = nn.Identity()  # cls concat
        self.layer5 = nn.Identity()  # pos_embed addition

        # ---------- Encoder Block 1 ----------
        self.layer6 = nn.LayerNorm(embed_dim)
        self.layer7 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer8 = nn.Identity()
        self.layer9 = nn.LayerNorm(embed_dim)
        self.layer10 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer11 = nn.Identity()

        # ---------- Encoder Block 2 ----------
        self.layer12 = nn.LayerNorm(embed_dim)
        self.layer13 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer14 = nn.Identity()
        self.layer15 = nn.LayerNorm(embed_dim)
        self.layer16 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer17 = nn.Identity()

        # ---------- Encoder Block 3 ----------
        self.layer18 = nn.LayerNorm(embed_dim)
        self.layer19 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer20 = nn.Identity()
        self.layer21 = nn.LayerNorm(embed_dim)
        self.layer22 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer23 = nn.Identity()

        # ---------- Encoder Block 4 ----------
        self.layer24 = nn.LayerNorm(embed_dim)
        self.layer25 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer26 = nn.Identity()
        self.layer27 = nn.LayerNorm(embed_dim)
        self.layer28 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer29 = nn.Identity()

        # ---------- Encoder Block 5 ----------
        self.layer30 = nn.LayerNorm(embed_dim)
        self.layer31 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer32 = nn.Identity()
        self.layer33 = nn.LayerNorm(embed_dim)
        self.layer34 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer35 = nn.Identity()

        # ---------- Encoder Block 6 ----------
        self.layer36 = nn.LayerNorm(embed_dim)
        self.layer37 = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer38 = nn.Identity()
        self.layer39 = nn.LayerNorm(embed_dim)
        self.layer40 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
        self.layer41 = nn.Identity()

        # ---------- Classification Head ----------
        self.layer42 = nn.LayerNorm(embed_dim)
        self.layer43 = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # Patch embedding
        out1 = self.layer1(x)              # [B, 128, 7, 7]
        out2 = self.layer2(out1)           # [B, 128, 49]
        out3 = out2.transpose(1, 2)        # [B, 49, 128]

        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 128]
        out4 = self.layer4(torch.cat([cls_token, out3], dim=1))  # [B, 50, 128]
        out5 = self.layer5(out4 + self.pos_embed)  # [B, 50, 128]

        # Block 1
        x = self.layer6(out5)
        x_attn, _ = self.layer7(x, x, x)
        x = self.layer8(x_attn + out5)
        x_mlp = self.layer10(self.layer9(x))
        x = self.layer11(x + x_mlp)

        # Block 2
        x_ = self.layer12(x)
        x_attn, _ = self.layer13(x_, x_, x_)
        x = self.layer14(x_attn + x)
        x_mlp = self.layer16(self.layer15(x))
        x = self.layer17(x + x_mlp)

        # Block 3
        x_ = self.layer18(x)
        x_attn, _ = self.layer19(x_, x_, x_)
        x = self.layer20(x_attn + x)
        x_mlp = self.layer22(self.layer21(x))
        x = self.layer23(x + x_mlp)

        # Block 4
        x_ = self.layer24(x)
        x_attn, _ = self.layer25(x_, x_, x_)
        x = self.layer26(x_attn + x)
        x_mlp = self.layer28(self.layer27(x))
        x = self.layer29(x + x_mlp)

        # Block 5
        x_ = self.layer30(x)
        x_attn, _ = self.layer31(x_, x_, x_)
        x = self.layer32(x_attn + x)
        x_mlp = self.layer34(self.layer33(x))
        x = self.layer35(x + x_mlp)

        # Block 6
        x_ = self.layer36(x)
        x_attn, _ = self.layer37(x_, x_, x_)
        x = self.layer38(x_attn + x)
        x_mlp = self.layer40(self.layer39(x))
        x = self.layer41(x + x_mlp)

        # Classification head
        out_cls = self.layer42(x[:, 0])
        out = self.layer43(out_cls)
        return out
