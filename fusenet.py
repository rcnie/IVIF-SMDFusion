import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.prelu2(out)
        return out

class Block1(nn.Module):
    def __init__(self, input, output):
        super(Block1, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0)
        self.ReLU = nn.ReLU()
        self.bath = nn.BatchNorm2d(output)

    def forward(self, x):
        out = self.conv(x)
        output0 = self.bath(out)
        output = self.ReLU(output0)
        return output

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, scales=[1, 2]):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0) for _ in scales
        ])

    def forward(self, x):
        attention_maps = []
        for scale, attention_layer in zip(self.scales, self.attention_layers):
            attention_map = attention_layer(x)
            attention_map = torch.sigmoid(torch.mean(attention_map, dim=1, keepdim=True))
            attention_maps.append(attention_map)

        out = sum(attention_maps) * x
        return out

class FCSA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        x = self.avgpool(input)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x

class UpSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.f(x)

class LSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, height=16, width=16):
        super(LSTM, self).__init__()
        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = input_channels
        self.kernel_size = kernel_size

        self.height = height
        self.width = width

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi1 = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi1 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Whf1 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc1 = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc1 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo1 = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who1 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxi2 = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Wxf2 = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Wxc2 = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo2 = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)

        self.Wci1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wcf1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wco1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wco2 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))


    def forward(self, outputA, outputB):
        B, C, H, W = outputA.size()
        A_reshape_tensor = outputA.reshape(B, C, self.height,self.width, -1)
        B_reshape_tensor = outputB.reshape(B, C, self.height,self.width, -1)

        A_blocks = torch.split(A_reshape_tensor, 1, dim=-1)
        outputA = torch.stack(A_blocks, dim=0).squeeze(1).squeeze(-1)

        B_blocks = torch.split(B_reshape_tensor, 1, dim=-1)
        outputB = torch.stack(B_blocks, dim=0).squeeze(1).squeeze(-1)

        ci = torch.sigmoid(self.Wxi1(outputA))
        cc = ci * torch.tanh(self.Wxc1(outputA))
        co = torch.sigmoid(self.Wxi1(outputA) + cc * self.Wco1)
        ch = co * torch.tanh(cc)

        df = torch.sigmoid(self.Wxf2(outputB) + self.Whf1(ch) + cc * self.Wcf1)
        di = torch.sigmoid(self.Wxi2(outputB) + self.Whi1(ch) + cc * self.Wci1)
        dc = di * torch.tanh(self.Wxc2(outputB) + self.Whc1(ch)) + df * cc
        do = torch.sigmoid(self.Wxi2(outputB) + self.Who1(ch) + dc * self.Wco2)
        dh = do * torch.tanh(dc)

        A_blocks = torch.split(dh, 1, dim=0)
        A_out = torch.stack(A_blocks, dim=-1)
        dh = A_out.reshape(B, C, H, W)
        return dh

class FUSENET(nn.Module):
    def __init__(self):
        super(FUSENET, self).__init__()

        self.L00 = Block(256, 128)
        self.L01 = Block1(256, 128)
        self.L02 = Block(128, 64)
        self.L03 = Block1(128, 64)
        self.L04 = Block(64, 32)
        self.L05 = Block1(64, 32)
        self.L06 = Block(32, 16)
        self.L07 = Block1(32, 16)
        self.L08 = Block(16, 1)

        self.L10 = Block(256, 128)
        self.L11 = Block(128, 64)
        self.L12 = Block(64, 32)
        self.L13 = Block(32, 16)

        self.L20 = Block(256, 128)
        self.L21 = Block(128, 64)
        self.L22 = Block(64, 32)
        self.L23 = Block(32, 16)

        self.u04 = UpSamplingBlock(128, 128)
        self.u05 = UpSamplingBlock(64, 64)
        self.u06 = UpSamplingBlock(32, 32)
        self.u07 = UpSamplingBlock(16, 16)

        self.u14 = UpSamplingBlock(128, 128)
        self.u15 = UpSamplingBlock(64, 64)
        self.u16 = UpSamplingBlock(32, 32)
        self.u17 = UpSamplingBlock(16, 16)

        self.fcsa00 = FCSA(128, 128)
        self.fcsa01 = FCSA(64, 64)
        self.fcsa02 = FCSA(32, 32)
        self.fcsa03 = FCSA(16, 16)
        self.fcsa10 = FCSA(128, 128)
        self.fcsa11 = FCSA(64, 64)
        self.fcsa12 = FCSA(32, 32)
        self.fcsa13 = FCSA(16, 16)
        self.fcsa20 = FCSA(128, 128)
        self.fcsa21 = FCSA(64, 64)
        self.fcsa22 = FCSA(32, 32)
        self.fcsa23 = FCSA(16, 16)

        self.MSA00 = MultiScaleAttention(128)
        self.MSA01 = MultiScaleAttention(64)
        self.MSA02 = MultiScaleAttention(32)
        self.MSA03 = MultiScaleAttention(16)
        self.MSA10 = MultiScaleAttention(128)
        self.MSA11 = MultiScaleAttention(64)
        self.MSA12 = MultiScaleAttention(32)
        self.MSA13 = MultiScaleAttention(16)

        self.LSTM1 = LSTM(input_channels=128, hidden_channels=128, kernel_size=1)
        self.LSTM2 = LSTM(64, 64, 1)
        self.LSTM3 = LSTM(32, 32, 1)
        self.LSTM4 = LSTM(16, 16, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d()

    def forward(self, tempA06, tempA04, tempA02, sA00, tempA14, tempA12, tempA10, sA10, tempA22, tempA21, tempA20, sA20,
                tempB06, tempB04, tempB02, sB00, tempB14, tempB12, tempB10, sB10, tempB22, tempB21, tempB20, sB20,
                outputA3, outputA2, outputA1, outputB3, outputB2, outputB1, imgA_V, imgB_V):

        self.dropout = nn.Dropout2d(0.1)

        tpA00 = self.L00(outputA1)
        tpA10 = self.L10(outputA2)
        tpA20 = self.L20(outputA3)
        tpB00 = self.L00(outputB1)
        tpB10 = self.L10(outputB2)
        tpB20 = self.L20(outputB3)

        sA25 = tpA20 + self.fcsa20(tempA22)
        sA15 = tpA10 + self.MSA10(self.u14(sA25)) + self.fcsa00(tempA14)
        sA05 = tpA00 + self.MSA00(self.u04(sA15)) + self.fcsa01(tempA06)

        sB25 = tpB20 + self.fcsa20(tempB22)
        sB15 = tpB10 + self.MSA10(self.u14(sB25)) + self.fcsa00(tempB14)
        sB05 = tpB00 + self.MSA00(self.u04(sB15)) + self.fcsa01(tempB06)

        EX01 = self.LSTM1(sA05, sB05)
        tpA01 = sA05 + EX01
        EX02 = self.LSTM1(sB05, sA05)
        tpB01 = sB05 + EX02

        tpA03 = self.L02(tpA01)
        tpA11 = self.L11(sA15)
        tpA21 = self.L21(sA25)
        tpB03 = self.L02(tpB01)
        tpB11 = self.L11(sB15)
        tpB21 = self.L21(sB25)

        sA26 = tpA21 + self.fcsa21(tempA21)
        sA16 = tpA11 + self.MSA11(self.u15(sA26)) + self.fcsa11(tempA12)
        sA06 = tpA03 + self.MSA01(self.u05(sA16)) + self.fcsa01(tempA04)
        sB26 = tpB21 + self.fcsa21(tempB21)
        sB16 = tpB11 + self.MSA11(self.u15(sB26)) + self.fcsa11(tempB12)
        sB06 = tpB03 + self.MSA01(self.u05(sB16)) + self.fcsa01(tempB04)
        EX03 = self.LSTM2(sA06, sB06)
        tpA04 = sA06 + EX03
        EX04 = self.LSTM2(sB06, sA06)
        tpB04 = sB06 + EX04

        tpA06 = self.L04(tpA04)
        tpA12 = self.L12(sA16)
        tpA22 = self.L22(sA26)
        tpB06 = self.L04(tpB04)
        tpB12 = self.L12(sB16)
        tpB22 = self.L22(sB26)

        sA27 = tpA22 + self.fcsa22(tempA20)
        sA17 = tpA12 + self.MSA12(self.u16(sA27)) + self.fcsa12(tempA10)
        sA07 = tpA06 + self.MSA02(self.u06(sA17)) + self.fcsa02(tempA02)
        sB27 = tpB22 + self.fcsa22(tempB20)
        sB17 = tpB12 + self.MSA12(self.u16(sB27)) + self.fcsa12(tempB10)
        sB07 = tpB06 + self.MSA02(self.u06(sB17)) + self.fcsa02(tempB02)

        EX05 = self.LSTM3(sA07, sB07)
        tpA07 = sA07 + EX05
        EX06 = self.LSTM3(sB07, sA07)
        tpB07 = sB07 + EX06

        tpA09 = self.L06(tpA07)
        tpA13 = self.L13(sA17)
        tpA23 = self.L23(sA27)
        tpB09 = self.L06(tpB07)
        tpB13 = self.L13(sB17)
        tpB23 = self.L23(sB27)

        sA28 = tpA23 + self.fcsa23(sA20)
        sA18 = tpA13 + self.MSA13(self.u17(sA28)) + self.fcsa13(sA10)
        sA08 = tpA09 + self.MSA03(self.u07(sA18)) + self.fcsa03(sA00)
        sB28 = tpB23 + self.fcsa23(sB20)
        sB18 = tpB13 + self.MSA13(self.u17(sB28)) + self.fcsa13(sB10)
        sB08 = tpB09 + self.MSA03(self.u07(sB18)) + self.fcsa03(sB00)

        EX07 = self.LSTM4(sA08, sB08)
        tpA010 = sA08 + EX07
        EX08 = self.LSTM4(sB08, sA08)
        tpB010 = sB08 + EX08

        tpA012 = self.L08(tpA010)
        tpB012 = self.L08(tpB010)

        input1 = self.sigmoid(tpA012)
        input2 = self.sigmoid(tpB012)

        weight_ir = input1
        weight_vis = input2
        Fusion = imgA_V * weight_ir + weight_vis * imgB_V

        return Fusion
