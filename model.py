import torch.nn as nn
import torch
import math
from block import ConvDownBlock


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=True):
        super(Discriminator, self).__init__()
        self.main = list()
        self.main.append(ConvDownBlock(3, 16, spec_norm, stride=2)) # 256 -> 128
        self.main.append(ConvDownBlock(16, 32, spec_norm, stride=2)) # 128 -> 64
        self.main.append(ConvDownBlock(32, 64, spec_norm, stride=2)) # 64 -> 32
        self.main.append(ConvDownBlock(64, 128, spec_norm, stride=2)) # 32 -> 16
        self.main.append(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

class ResBlockNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockNet, self).__init__()
        self.main = list()
        self.main.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

class Encoder(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channels=3, spec_norm=False, LR=0.2):
        super(Encoder, self).__init__()

        self.layer1 = ConvDownBlock(in_channels, 16, spec_norm, LR=LR)
        self.layer2 = ConvDownBlock(16, 16, spec_norm, LR=LR)
        self.layer3 = ConvDownBlock(16, 32, spec_norm, stride=2, LR=LR)
        self.layer4 = ConvDownBlock(32, 32, spec_norm, LR=LR)
        self.layer5 = ConvDownBlock(32, 64, spec_norm, stride=2, LR=LR)
        self.layer6 = ConvDownBlock(64, 64, spec_norm, LR=LR)
        self.layer7 = ConvDownBlock(64, 128, spec_norm, stride=2, LR=LR)
        self.layer8 = ConvDownBlock(128, 128, spec_norm, LR=LR)
        self.layer9 = ConvDownBlock(128, 256, spec_norm, stride=2, LR=LR)
        self.layer10 = ConvDownBlock(256, 256, spec_norm, LR=LR)
        self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):

        feature_map1 = self.layer1(x)
        feature_map2 = self.layer2(feature_map1)
        feature_map3 = self.layer3(feature_map2)
        feature_map4 = self.layer4(feature_map3)
        feature_map5 = self.layer5(feature_map4)
        feature_map6 = self.layer6(feature_map5)
        feature_map7 = self.layer7(feature_map6)
        feature_map8 = self.layer8(feature_map7)
        feature_map9 = self.layer9(feature_map8)
        feature_map10 = self.layer10(feature_map9)

        down_feature_map1 = self.down_sampling(feature_map1)
        down_feature_map2 = self.down_sampling(feature_map2)
        down_feature_map3 = self.down_sampling(feature_map3)
        down_feature_map4 = self.down_sampling(feature_map4)
        down_feature_map5 = self.down_sampling(feature_map5)
        down_feature_map6 = self.down_sampling(feature_map6)
        down_feature_map7 = self.down_sampling(feature_map7)
        down_feature_map8 = self.down_sampling(feature_map8)

        """
        print("feature_map1.size : ", self.down_sampling(feature_map1).size())  
        print("feature_map2.size : ", self.down_sampling(feature_map2).size()) 
        print("feature_map3.size : ", self.down_sampling(feature_map3).size())
        print("feature_map4.size : ", self.down_sampling(feature_map4).size())
        print("feature_map5.size : ", self.down_sampling(feature_map5).size())
        print("feature_map6.size : ", self.down_sampling(feature_map6).size())
        print("feature_map7.size : ", self.down_sampling(feature_map7).size())
        print("feature_map8.size : ", self.down_sampling(feature_map8).size())
        print("feature_map9.size : ", feature_map9.size())
        print("feature_map10.size : ", feature_map10.size())
        """

        output = torch.cat([down_feature_map1,
                            down_feature_map2,
                            down_feature_map3,
                            down_feature_map4,
                            down_feature_map5,
                            down_feature_map6,
                            down_feature_map7,
                            down_feature_map8,
                            feature_map9,
                            feature_map10,
                            ], dim=1)

        feature_list = [down_feature_map1,
         down_feature_map2,
         down_feature_map3,
         down_feature_map4,
         down_feature_map5,
         down_feature_map6,
         down_feature_map7,
         down_feature_map8,
         feature_map9,
         feature_map10,
         ]
        #print('output.size : ', output.size()) # output.size :  torch.Size([2, 992, 16, 16])
        b, ch, h, w = output.size()
        #output = output.reshape((b, ch, h * w)) # output.size :  torch.Size([2, 992, 256])
        output = output.reshape((b, h * w, ch)) # output.size :  torch.Size([2, 256, 992])
        #print('output.size : ', output.size())
        return output, feature_list

class Decoder(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channels=3, spec_norm=False, LR=0.2):
        super(Decoder, self).__init__()

    def forward(self, x):
        return x


class SCFT_Moudle(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self):
        super(SCFT_Moudle, self).__init__()
        self.w_q = nn.Linear(992, 992)
        self.w_k = nn.Linear(992, 992)
        self.w_v = nn.Linear(992, 992)
        self.scailing_factor = math.sqrt(992)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, v_s, v_r):

        #print('v_s.size() : ', v_s.size()) # v_s.size() :  torch.Size([2, 256, 992])
        #print('v_r.size() : ', v_r.size()) # v_r.size() :  torch.Size([2, 256, 992])
        q_result = self.w_q(v_s)
        k_result = self.w_k(v_r)
        v_result = self.w_v(v_r)

        #print('q_result : ', q_result.size()) # q_result :  torch.Size([2, 256, 992])
        #print('k_result : ', k_result.size()) # k_result :  torch.Size([2, 256, 992])
        k_result = k_result.permute(0, 2, 1)
        #print('k_result : ', k_result.size()) # k_result :  torch.Size([2, 992, 256])
        attention_map = torch.bmm(q_result, k_result)
        #print('attention_map : ', attention_map.size()) # attention_map :  torch.Size([2, 256, 256])
        attention_map = self.softmax(attention_map) / self.scailing_factor
        v_star = torch.bmm(attention_map, v_result)
        #print('v_star.size() : ', v_star.size()) # v_star.size() :  torch.Size([2, 256, 992])
        """
        *debug example
        soft_max = nn.Softmax(dim=2)
        t = torch.randn((2,5,5))
        softed_t = soft_max(t)
        print(softed_t)
        print(torch.sum(softed_t[0, 0]))
        """
        v_sum = (v_s + v_star).permute(0, 2, 1)
        #print('v_sum.size() : ', v_sum.size()) # v_sum.size() :  torch.Size([2, 992, 256])
        b, ch, hw = v_sum.size()
        v_sum = v_sum.reshape((b, ch, 16, 16))
        #print('v_sum.size() : ', v_sum.size())  # v_sum.size() :  torch.Size([2, 992, 16, 16])
        return v_sum

class Combined_Model(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channels=3, spec_norm=False, LR=0.2):
        super(Combined_Model, self).__init__()

    def forward(self, x):
        return x