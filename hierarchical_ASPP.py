import torch.nn as nn
import torch
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):                  #kernel_size = [7,7]   [11,11]  [15,15]          [3,3] [7,7][11,11]
        super(GCN, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))  # 左kx1卷积
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))  # 左1xk卷积
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))  # 右1xk卷积
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))  # 右kx1卷积

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r  # sum操作
        return x
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),     #设置了dilation就是空洞卷积  卷积核的大小可以自己设置
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:              #这里继承的事nn.sequential 所以初始化下面定义的层就是sequential的形式，这里直接遍历self就行
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class H_ASPP(nn.Module):
    def __init__(self, in_channels, k):    #  k= [[[3,3],[7,7]], [[5,5],[9,9]], [[7,7],[11,11]]]
        super(H_ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            ))      #后面就不加入BN和Relu了

        rates = tuple(k)
        for rate in rates:
            modules.append(nn.Sequential(GCN(in_channels, out_channels, rate[0]),GCN(in_channels, out_channels, rate[1])))
        self.convs = nn.ModuleList(modules)    #用list在包装一下

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []   #这个列表用于存放最后的输出
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# many are borrowed from https://github.com/ycszen/pytorch-ss/blob/master/gcn.py



class H_ASPP_high(nn.Module):
    def __init__(self, in_channels, k):    #  k= [[1,2], [1,3], [1,4]]
        super(H_ASPP_high, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            ))      #后面就不加入BN和Relu了

        rates = tuple(k)
        for rate in rates:
            modules.append(nn.Sequential(ASPPConv(in_channels, out_channels, rate[0]),ASPPConv(in_channels, out_channels, rate[1])))
        self.convs = nn.ModuleList(modules)    #用list在包装一下

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)