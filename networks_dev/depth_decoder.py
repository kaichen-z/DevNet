import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1, bias=bias)
    def forward(self, x):
        out = self.conv(x)
        return out

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'pointwise'), Conv1x1(in_planes if (i == 0) else out_planes, out_planes, False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'pointwise'))(top)
            x = top + x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes,in_planes // ratio, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_planes // ratio, in_planes, bias = False))
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b,c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature

class Attention_Module1(nn.Module):
    def __init__(self, high_feature_channel, output_channel = None):
        super(Attention_Module1, self).__init__()
        in_channel = high_feature_channel 
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        self.conv_se = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 1, stride = 1, padding = 0 )
        self.relu = nn.ReLU(inplace = True)
    def forward(self, high_features):
        features = high_features
        features = self.ca(features)
        return self.relu(self.conv_se(features))

class Attention_Module3(nn.Module):
    def __init__(self, high_feature_channel, output_channel = None):
        super(Attention_Module3, self).__init__()
        in_channel = high_feature_channel 
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        self.conv_se = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1 )
        self.relu = nn.ReLU(inplace = True)
    def forward(self, high_features):
        features = high_features
        features = self.ca(features)
        return self.relu(self.conv_se(features))

def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, embedder, embedder_out_dim, output_channels, use_alpha):
        super(DepthDecoder, self).__init__()
        bottleneck = [512, 256, 256, 256, 256, output_channels]
        stage = 4
        self.do = nn.Dropout(p=0.5)
        self.embedder = embedder
        self.E = embedder_out_dim
        self.use_alpha = use_alpha
        self.sigma_dropout_rate = 0.0

        self.attention1 = False
        self.attention2 = False
        self.attention3 = True
        """
        Structure: 
        x4 = self.reduce4(l4) # 1
        # Merge
        x4 = self.iconv4(x4) # 2
        x4 = self.crp4(x4)
        x4 = self.merge4(x4) # 3
        """

        self.skip_link = True
        self.p1, self.p2, self.p3 = [1.], [0.5], [0.1]
        #self.p1, self.p2, self.p3 = [1.], [1.], [1.]

        self.elu = 1
        self.soft_plus = False
        if self.elu == 0:
            self.activation_func = nn.ELU()
        elif self.elu == 1:
            self.activation_func = nn.LeakyReLU()
        elif self.elu == 2:
            self.activation_func = nn.ReLU()

        if not self.attention1:
            # For new feature
            self.reduce4 = Conv1x1(num_ch_enc[4], bottleneck[0], bias=False)
            self.reduce3 = Conv1x1(num_ch_enc[3], bottleneck[1], bias=False)
            self.reduce2 = Conv1x1(num_ch_enc[2], bottleneck[2], bias=False)
            self.reduce1 = Conv1x1(num_ch_enc[1], bottleneck[3], bias=False)
            self.reduce0 = Conv1x1(num_ch_enc[0], bottleneck[4], bias=False)
        else:
            self.reduce4 = Attention_Module1(num_ch_enc[4], bottleneck[0])
            self.reduce3 = Attention_Module1(num_ch_enc[3], bottleneck[1])
            self.reduce2 = Attention_Module1(num_ch_enc[2], bottleneck[2])
            self.reduce1 = Attention_Module1(num_ch_enc[1], bottleneck[3])       
            self.reduce0 = Attention_Module1(num_ch_enc[0], bottleneck[4])

        if not self.skip_link:
            if not self.attention2:
                self.iconv4 = Conv3x3(bottleneck[0], bottleneck[1])
                self.iconv3 = Conv3x3(bottleneck[1]*2, bottleneck[2])
                self.iconv2 = Conv3x3(bottleneck[2]*2, bottleneck[3])
                self.iconv1 = Conv3x3(bottleneck[3]*2, bottleneck[4])
                self.iconv0 = Conv3x3(bottleneck[4]*2, bottleneck[5])
            else:
                self.iconv4 = Attention_Module3(bottleneck[0], bottleneck[1])
                self.iconv3 = Attention_Module3(bottleneck[1]*2, bottleneck[2])
                self.iconv2 = Attention_Module3(bottleneck[2]*2, bottleneck[3])
                self.iconv1 = Attention_Module3(bottleneck[3]*2, bottleneck[4])
                self.iconv0 = Attention_Module3(bottleneck[4]*2, bottleneck[5])
        else:
            if not self.attention2:
                self.iconv4 = Conv3x3(bottleneck[0], bottleneck[1])
                self.iconv3 = Conv3x3(bottleneck[1]*2+output_channels, bottleneck[2])
                self.iconv2 = Conv3x3(bottleneck[2]*2+output_channels, bottleneck[3])
                self.iconv1 = Conv3x3(bottleneck[3]*2+output_channels, bottleneck[4])
                self.iconv0 = Conv3x3(bottleneck[4]*2+output_channels, bottleneck[5])
            else:
                self.iconv4 = Attention_Module3(bottleneck[0], bottleneck[1])
                self.iconv3 = Attention_Module3(bottleneck[1]*2+output_channels, bottleneck[2])
                self.iconv2 = Attention_Module3(bottleneck[2]*2+output_channels, bottleneck[3])
                self.iconv1 = Attention_Module3(bottleneck[3]*2+output_channels, bottleneck[4])
                self.iconv0 = Conv3x3(bottleneck[4]*2+output_channels, bottleneck[5])
        """
        self.crp4 = self._make_crp(bottleneck[1], bottleneck[1], stage)
        self.crp3 = self._make_crp(bottleneck[2], bottleneck[2], stage)
        self.crp2 = self._make_crp(bottleneck[3], bottleneck[3], stage)
        self.crp1 = self._make_crp(bottleneck[4], bottleneck[4], stage)
        self.crp0 = self._make_crp(bottleneck[5], bottleneck[5], stage)
        """
        if not self.attention3:
            self.merge4 = Conv3x3(bottleneck[1], bottleneck[1])
            self.merge3 = Conv3x3(bottleneck[2], bottleneck[2])
            self.merge2 = Conv3x3(bottleneck[3], bottleneck[3])
            self.merge1 = Conv3x3(bottleneck[4], bottleneck[4])
            self.merge0 = Conv3x3(bottleneck[5], bottleneck[5])
        else:
            # feature fusion
            self.merge4 = Attention_Module3(bottleneck[1], bottleneck[1])
            self.merge3 = Attention_Module3(bottleneck[2], bottleneck[2])
            self.merge2 = Attention_Module3(bottleneck[3], bottleneck[3])
            self.merge1 = Attention_Module3(bottleneck[4], bottleneck[4])
            self.merge0 = Attention_Module3(bottleneck[5], bottleneck[5])

        # disp
        if not self.use_alpha:
            self.disp4 = nn.Sequential(Conv3x3(bottleneck[1], output_channels), nn.ReLU())
            self.disp3 = nn.Sequential(Conv3x3(bottleneck[2], output_channels), nn.ReLU())
            self.disp2 = nn.Sequential(Conv3x3(bottleneck[3], output_channels), nn.ReLU())
            self.disp1 = nn.Sequential(Conv3x3(bottleneck[4], output_channels), nn.ReLU())
            self.disp0 = nn.Sequential(Conv3x3(bottleneck[5], output_channels), nn.ReLU())
        else:
            self.disp4 = nn.Sequential(Conv3x3(bottleneck[1], output_channels), nn.Sigmoid())
            self.disp3 = nn.Sequential(Conv3x3(bottleneck[2], output_channels), nn.Sigmoid())
            self.disp2 = nn.Sequential(Conv3x3(bottleneck[3], output_channels), nn.Sigmoid())
            self.disp1 = nn.Sequential(Conv3x3(bottleneck[4], output_channels), nn.Sigmoid())
            self.disp0 = nn.Sequential(Conv3x3(bottleneck[5], output_channels), nn.Sigmoid())       

        if self.soft_plus:
            self.disp4 = nn.Sequential(Conv3x3(bottleneck[1], output_channels), nn.Softplus())
            self.disp3 = nn.Sequential(Conv3x3(bottleneck[2], output_channels), nn.Softplus())
            self.disp2 = nn.Sequential(Conv3x3(bottleneck[3], output_channels), nn.Softplus())
            self.disp1 = nn.Sequential(Conv3x3(bottleneck[4], output_channels), nn.Softplus())
            self.disp0 = nn.Sequential(Conv3x3(bottleneck[5], output_channels), nn.Softplus())
        
    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, input_features, disparity):
        B, S = disparity.size()
        disp_list = self.embedder(disparity.reshape(B * S, 1)).unsqueeze(2).unsqueeze(3) 

        self.outputs = {}
        l0, l1, l2, l3, l4 = input_features
        if self.training:
            l4, l3 = self.do(l4), self.do(l3)

        x4 = self.reduce4(l4)
        x4 = self.iconv4(x4)
        x4 = self.activation_func(x4)
        #x4 = self.crp4(x4)
        x4 = self.merge4(x4)
        x4 = self.activation_func(x4)
        x4 = upsample(x4)
        disp4 = self.disp4(x4)

        x3 = self.reduce3(l3)
        _, _, H_feat, W_feat = x3.size()
        x3 = torch.cat((self.p1[0]*x3, self.p2[0]*x4, self.p3[0]*disp4), 1)
        x3 = self.iconv3(x3)
        x3 = self.activation_func(x3)
        #x3 = self.crp3(x3)
        x3 = self.merge3(x3)
        x3 = self.activation_func(x3)
        x3 = upsample(x3)
        disp3 = self.disp3(x3)
        if self.sigma_dropout_rate > 0.0 and self.training:
            disp3 = F.dropout2d(disp3, p=self.sigma_dropout_rate)

        x2 = self.reduce2(l2)
        _, _, H_feat, W_feat = x2.size()
        if not self.skip_link:
            x2 = torch.cat((self.p1[0]*x2, self.p2[0]*x3), 1)
        else:
            x2 = torch.cat((self.p1[0]*x2, self.p2[0]*x3, self.p3[0]*disp3), 1)
        x2 = self.iconv2(x2)
        x2 = self.activation_func(x2)
        #x2 = self.crp2(x2)
        x2 = self.merge2(x2)
        x2 = self.activation_func(x2)
        x2 = upsample(x2)
        disp2 = self.disp2(x2)
        if self.sigma_dropout_rate > 0.0 and self.training:
            disp2 = F.dropout2d(disp2, p=self.sigma_dropout_rate)

        x1 = self.reduce1(l1)
        _, _, H_feat, W_feat = x1.size()
        if not self.skip_link:
            x1 = torch.cat((self.p1[0]*x1, self.p2[0]*x2), 1)
        else:
            x1 = torch.cat((self.p1[0]*x1, self.p2[0]*x2, self.p3[0]*disp2), 1)
        x1 = self.iconv1(x1)
        x1 = self.activation_func(x1)
        #x1 = self.crp1(x1)
        x1 = self.merge1(x1)
        x1 = self.activation_func(x1)
        x1 = upsample(x1)
        disp1 = self.disp1(x1)
        if self.sigma_dropout_rate > 0.0 and self.training:
            disp1 = F.dropout2d(disp1, p=self.sigma_dropout_rate)

        x0 = self.reduce0(l0)
        _, _, H_feat, W_feat = x0.size()
        if not self.skip_link:
            x0 = torch.cat((self.p1[0]*x0, self.p2[0]*x1), 1)
        else:
            x0 = torch.cat((self.p1[0]*x0, self.p2[0]*x1, self.p3[0]*disp1), 1)
        x0 = self.iconv0(x0)
        x0 = self.activation_func(x0)
        #x0 = self.crp0(x0)
        x0 = self.merge0(x0)
        x0 = self.activation_func(x0)
        x0 = upsample(x0)
        disp0 = self.disp0(x0)
        if self.sigma_dropout_rate > 0.0 and self.training:
            disp0 = F.dropout2d(disp0, p=self.sigma_dropout_rate)

        H_mpi, W_mpi = disp3.size(2), disp3.size(3)
        if not self.use_alpha:
            self.outputs[("disp", 3)] = disp3.view(B, S, H_mpi, W_mpi).unsqueeze(2)  + 1e-8 
        elif self.use_alpha:
            self.outputs[("disp", 3)] = disp3.view(B, S, H_mpi, W_mpi).unsqueeze(2)

        H_mpi, W_mpi = disp2.size(2), disp2.size(3)
        if not self.use_alpha:
            self.outputs[("disp", 2)] = disp2.view(B, S, H_mpi, W_mpi).unsqueeze(2)  + 1e-8 
        elif self.use_alpha:
            self.outputs[("disp", 2)] = disp2.view(B, S, H_mpi, W_mpi).unsqueeze(2)

        H_mpi, W_mpi = disp1.size(2), disp1.size(3)
        if not self.use_alpha:
            self.outputs[("disp", 1)] = disp1.view(B, S, H_mpi, W_mpi).unsqueeze(2)  + 1e-8 
        elif self.use_alpha:
            self.outputs[("disp", 1)] = disp1.view(B, S, H_mpi, W_mpi).unsqueeze(2)
        
        H_mpi, W_mpi = disp0.size(2), disp0.size(3)
        if not self.use_alpha:
            self.outputs[("disp", 0)] = disp0.view(B, S, H_mpi, W_mpi).unsqueeze(2)  + 1e-8 
        elif self.use_alpha:
            self.outputs[("disp", 0)] = disp0.view(B, S, H_mpi, W_mpi).unsqueeze(2)
        return self.outputs