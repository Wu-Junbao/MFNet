import torch.nn as nn
import torch
import torch.nn.functional as F

class Attfuse2(nn.Module):
    def __init__(self,in_chanel):
        super(Attfuse2, self).__init__()
        self.conv_c = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Conv2d(in_chanel*2,in_chanel*2,kernel_size=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_chanel*2,in_chanel,kernel_size=1),
                                    nn.Sigmoid())   #得到通道权重
        self.conv_s = nn.Sequential(nn.Conv2d(in_chanel*2, 1,kernel_size=1),
                                    nn.Sigmoid())    #得到空间的权重
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    # 网络权重的初始化方式是否也要定义一下





    def forward(self, x1, x2):
        size = x1.shape[-2:]
        x2_in = F.interpolate(x2, size=size, mode='bilinear', align_corners=False)
        fuse = torch.cat((x1, x2_in), dim=1)    #这里其实能改成直接相乘的   x1 * x2
        w_c = self.conv_c(fuse)
        w_s = self.conv_s(fuse)
        x_out1 = x1 * w_s
        x_out2 = x2 * w_c             #先进行通道 注意力之后在进行下一步线性插值  想先增强在插入噪声  会不会更好一些
        x_out2 = self.upsample(x_out2)
        return x_out1 + x_out2




# fuse=Attfuse2()
# fuse(x1)




#
# class Attfuse3(nn.Module):    #就做两个的交互吧 ，就不使用三个的交互了 另一个残差的结构别给破坏掉
#     def __init__(self, ):
#         pass