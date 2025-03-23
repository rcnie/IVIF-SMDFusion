import torch.nn as nn

class Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU()

        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
            self.ReLU = nn.ReLU()

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
            x = self.ReLU(x)
            output = fx + x
            output = self.ReLU(output)
        return output

class Block1(nn.Module):
    def __init__(self, input, output):
        super(Block1, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0)
        self.ReLU = nn.ReLU()
        self.bath = nn.BatchNorm2d(output)
    def forward (self, x):
            out = self.conv(x)
            # output0 = self.bath(out)
            output = self.ReLU(out)
            return output

class DownSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.f(x)

class UpSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.UpsamplingNearest2d(scale_factor = 2),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.f(x)

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.c01 = Block(1, 16)
        self.c02 = Block1(32, 16)
        self.c03 = Block(16, 32)
        self.c04 = Block1(64, 32,)
        self.c05 = Block(32, 64)
        self.c06 = Block(128, 64)
        self.c07 = Block(64, 128)
        self.c08 = Block(256, 128)
        self.c09 = Block(128, 256)

        self.c13 = Block(16, 32)
        self.c15 = Block(32, 64)
        self.c16 = Block(64, 128)
        self.c17 = Block(128, 256)

        self.c23 = Block(16, 32)
        self.c25 = Block(32, 64)
        self.c26 = Block(64, 128)
        self.c27 = Block(128, 256)

        self.d00 = DownSamplingBlock(16, 16)
        self.d01 = DownSamplingBlock(32, 32)
        self.d02 = DownSamplingBlock(64, 64)
        self.d03 = DownSamplingBlock(128, 128)
        self.d10 = DownSamplingBlock(16, 16)
        self.d11 = DownSamplingBlock(32, 32)
        self.d12 = DownSamplingBlock(64, 64)
        self.d13 = DownSamplingBlock(128, 128)

        self.u00 = UpSamplingBlock(32, 32)
        self.u01 = UpSamplingBlock(64, 64)
        self.u02 = UpSamplingBlock(128, 128)
        self.u03 = UpSamplingBlock(256, 256)
        self.u10 = UpSamplingBlock(32, 32)
        self.u11 = UpSamplingBlock(64, 64)
        self.u12 = UpSamplingBlock(128, 128)
        self.u13 = UpSamplingBlock(256, 256)

        self.Sigmoid = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, imgeA, imgeB):

        sA00 = self.c01(imgeA)
        sB00 = self.c01(imgeB)
        sA10 = self.d00(sA00)
        sB10 = self.d00(sB00)
        sA20 = self.d10(sA10)
        sB20 = self.d10(sB10)

        tempA01 = sA00
        tempA02 = self.c03(tempA01)

        tempB01 = sB00
        tempB02 = self.c03(tempB01)
        tempA10 = self.c13(sA10)
        tempA20 = self.c23(sA20)
        tempB10 = self.c13(sB10)
        tempB20 = self.c23(sB20)

        sA01 = tempA02
        sB01 = tempB02
        sA11 = tempA10 + self.d01(tempA02)
        sB11 = tempB10 + self.d01(tempB02)
        sA21 = tempA20 + self.d11(tempA10 + self.d01(tempA02))
        sB21 = tempB20 + self.d11(tempB10 + self.d01(tempB02))

        tempA03 = sA01
        tempA04 = self.c05(tempA03)
        tempB03 = sB01
        tempB04 = self.c05(tempB03)
        tempA12 = self.c15(sA11)
        tempA21 = self.c25(sA21)
        tempB12 = self.c15(sB11)
        tempB21 = self.c25(sB21)

        sA02 = tempA04
        sB02 = tempB04
        sA12 = tempA12 + self.d02(tempA04)
        sB12 = tempB12 + self.d02(tempB04)
        sA22 = tempA21 + self.d12(tempA12 + self.d02(tempA04))
        sB22 = tempB21 + self.d12(tempB12 + self.d02(tempB04))

        tempA05 = sA02
        tempA06 = self.c07(tempA05)
        tempB05 = sB02
        tempB06 = self.c07(tempB05)
        tempA14 = self.c16(sA12)
        tempA22 = self.c26(sA22)
        tempB14 = self.c16(sB12)
        tempB22 = self.c26(sB22)

        sA03 = tempA06
        sB03 = tempB06
        sA13 = tempA14 + self.d03(tempA06)
        sB13 = tempB14 + self.d03(tempB06)
        sA23 = tempA22 + self.d13(tempA14 + self.d03(tempA06))
        sB23 = tempB22 + self.d13(tempB14 + self.d03(tempB06))

        tempA07 = sA03
        tempB07 = sB03
        tempA08 = self.c09(tempA07)
        tempB08 = self.c09(tempB07)
        sA24 = self.c27(sA23)
        sA14 = self.c17(sA13) + self.u13(sA24)
        sA04 = tempA08 + self.u03(sA14)
        sB24 = self.c27(sB23)
        sB14 = self.c17(sB13) + self.u13(sB24)
        sB04 = tempB08 + self.u03(sB14)

        outputA = self.decoder(sA04)
        outputB = self.decoder(sB04)
        outputA1 = tempA08
        outputB1 = tempB08
        outputA2 = self.c17(sA13)
        outputB2 = self.c17(sB13)
        outputA3 = sA24
        outputB3 = sB24

        return tempA06, tempA04, tempA02, sA00, tempA14, tempA12, tempA10, sA10, tempA22, tempA21, tempA20, sA20, \
               tempB06, tempB04, tempB02, sB00, tempB14, tempB12, tempB10, sB10, tempB22, tempB21, tempB20, sB20, \
               outputA3, outputA2, outputA1, outputB3, outputB2, outputB1, outputA, outputB