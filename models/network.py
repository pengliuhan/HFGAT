import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.register import register
from models.base_model import *

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x):
        self.elements = x.shape[1] * x.shape[2] * x.shape[3]
        self.last_jac = self.elements / 4 * np.log(1/16.)
        out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
        out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
        return out

class InvHaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(InvHaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x):
        self.elements = x.shape[1] * x.shape[2] * x.shape[3]
        self.last_jac = self.elements / 4 * np.log(16.)
        out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
        return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_channel//4, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        return self.deconv(x)

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class FreqAwareTransformer(nn.Module):
    def __init__(self, in_channel, H, W, drop_path):
        super().__init__()
        self.conv2 = nn.Sequential(
            HaarDownsampling(in_channel),
            nn.BatchNorm2d(in_channel * 4),
            nn.Conv2d(in_channel * 4, in_channel, kernel_size=3, stride=1, groups=4, padding=1, bias=False),
        )
        self.encoder1 = BasicUformerLayer(dim=in_channel, input_resolution=(H, W), depth=1, num_heads=4,
                                          drop_path=drop_path, token_mlp='leff', norm_layer=nn.BatchNorm2d)
        self.encoder2 = BasicUformerLayer(dim=in_channel, input_resolution=(H, W), depth=1, num_heads=1,
                                          drop_path=drop_path, token_mlp='leff', norm_layer=nn.BatchNorm2d)
        self.inverse = nn.Sequential(
            InvHaarDownsampling(in_channel//4),
            nn.Conv2d(in_channel//4, in_channel, kernel_size=3, stride=1, groups=1, padding=1, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv2(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        x = self.encoder1(x, H//2, W//2, ln=False)
        x = self.encoder2(x, H//2, W//2, ln=False)
        x = x.transpose(1, 2).contiguous().view(B, C, H//2, W//2)
        x = self.inverse(x)
        return x

class HFGAT(nn.Module):
    def __init__(self, embed_dim=92, depths=[2, 2, 2, 2, 2, 2, 2], drop_path_rate=0.1):
        super().__init__()
        H=64
        W=64
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[3]
        dec_dpr = enc_dpr[::-1]

        self.encoderlayer_0 = FreqAwareTransformer(embed_dim, H, W, enc_dpr[sum(depths[:0]):sum(depths[:1])])
        self.dowsample_0 = Downsample(embed_dim, embed_dim)
        self.encoderlayer_1 = FreqAwareTransformer(embed_dim, H//2, W//2, enc_dpr[sum(depths[:1]):sum(depths[:2])])
        self.dowsample_1 = Downsample(embed_dim, embed_dim)
        self.encoderlayer_2 = FreqAwareTransformer(embed_dim, H // (2 ** 2), W // (2 ** 2), enc_dpr[sum(depths[:2]):sum(depths[:3])])
        self.dowsample_2 = Downsample(embed_dim, embed_dim)
        self.conv =FreqAwareTransformer(embed_dim, H // (2 ** 3), W // (2 ** 3), conv_dpr)

        self.upsample_0 = Upsample(embed_dim, embed_dim)
        self.skff_0 = SKFF(in_channels=embed_dim, height=2, reduction=8)
        self.decoderlayer_0 = FreqAwareTransformer(embed_dim, H // (2 ** 2), W // (2 ** 2), dec_dpr[:depths[4]])
        self.upsample_1 = Upsample(embed_dim, embed_dim)
        self.skff_1 = SKFF(in_channels=embed_dim, height=2, reduction=8)
        self.decoderlayer_1 = FreqAwareTransformer(embed_dim, H // 2, W // 2, dec_dpr[sum(depths[4:5]):sum(depths[4:6])])
        self.upsample_2 = Upsample(embed_dim, embed_dim)
        self.skff_2 = SKFF(in_channels=embed_dim, height=2, reduction=8)
        self.decoderlayer_2 = FreqAwareTransformer(embed_dim, H, W, dec_dpr[sum(depths[4:6]):sum(depths[4:7])])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, freq):
        conv0 = self.encoderlayer_0(freq)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)

        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)
        conv4 = self.conv(pool2)

        up0 = self.upsample_0(conv4)
        deconv0 = self.skff_0([up0, conv2])
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = self.skff_1([up1, conv1])
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = self.skff_2([up2, conv0])
        deconv2 = self.decoderlayer_2(deconv2)
        return deconv2

@register('network')
class Network(nn.Module):
    def __init__(self, in_nc=21):
        super().__init__()
        dim=96
        self.input_conv = nn.Conv2d(in_nc, dim, kernel_size=3, stride=1, groups=1, padding=1)
        self.ffnet = HFGAT(embed_dim=dim)
        self.output = nn.Conv2d(dim, 3, 3, padding=3 // 2)

    def forward(self, inputs):
        B, T, C, H, W = inputs.shape
        spatial = self.input_conv(inputs.view(B, C * 7, H, W))
        spatial = self.ffnet(spatial)
        img = inputs[:,3,:,:,:]+self.output(spatial)
        return img

if __name__ == "__main__":
    torch.cuda.set_device(4)
    net = Network().cuda()
    from thop import profile

    input = torch.randn(1, 7, 3, 256, 256).cuda()
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))


