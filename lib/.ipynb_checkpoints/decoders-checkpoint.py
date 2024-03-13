import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.convnextv2 import *
import torchvision
#from lib.convnext import Block


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x
    
class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
	        nn.GELU()
        )

    def forward(self,x):
        x = self.up(x)
        return x

        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int),
            #nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int),
            #nn.GELU(),
            nn.Dropout(0.3)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.GELU()
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1+x1)
        
        psi = self.psi(psi)

        return x*psi

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.GELU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x) #torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    

class CASCADE_Cat(nn.Module):
    def __init__(self, channels=[512,320,128,64],depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0.,):
        super(CASCADE_Cat,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
        
        
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=2*channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2*channels[1])
        self.CA2 = ChannelAttention(2*channels[2])
        self.CA1 = ChannelAttention(2*channels[3])
        
        self.SA = SpatialAttention()

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        print(999999999999999999999999999999999999999999)
        print(d4.shape)
        d4 = self.stages[3](d4)
        #d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # Concat 3
        d3 = torch.cat((x3,d3),dim=1)
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        #d3 = self.ConvBlock3(d3)
        d3 = self.stages[2](d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # Concat 2
        d2 = torch.cat((x2,d2),dim=1)
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        #d2 = self.ConvBlock2(d2)
        d2 = self.stages[2](d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # Concat 1
        d1 = torch.cat((x1,d1),dim=1)
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        #d1 = self.ConvBlock1(d1)
        d1 = self.stages[0](d1)
        return d4, d3, d2, d1

class Wide_Focus(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.layer_dilation2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=2),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.layer_deepth = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1,padding="same" ,bias=True),
            nn.GELU()
        )
        self.layer_dilation3 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=3),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer_dilation2(x)
        x3 = self.layer_deepth(x)
        added = x1 + x2 + x3
        x_out = self.layer4(added)
        return x_out

class SE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.GELU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class CASCADE_Add(nn.Module):
    def __init__(self, channels=[512,320,128,64],depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,layer_scale_init_value=1e-6):
        super(CASCADE_Add,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.AG44 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=channels[3])
        self.AG33 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[2])        
        self.AG22 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[1])
        self.AG11 = Attention_block(F_g=channels[0],F_l=channels[0],F_int=int(channels[0]/2))

        
        self.SA = SpatialAttention()
        
        self.SE0 = SE(channels[0])
        self.SE1 = SE(channels[1])
        self.SE2 = SE(channels[2])
        self.SE3 = SE(channels[3])
        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur]) ]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.concat1 = conv_block(ch_in=768,ch_out=384)
        self.concat2 = conv_block(ch_in=384,ch_out=192)
        #self.concat2 = nn.Conv2d(in_channels= 384, out_channels= 192, kernel_size=3,stride=1,padding=1,bias=True)
        self.concat3 = conv_block(ch_in=192, ch_out=96)
       # self.concat3 = nn.Conv2d(in_channels= 192, out_channels= 96, kernel_size=3,stride=1,padding=1,bias=True)

        self.wide1 = Wide_Focus(in_channels= 768,out_channels= 768)
        self.wide2 = Wide_Focus(in_channels=384, out_channels=384)
        self.wide3 = Wide_Focus(in_channels=192, out_channels= 192)
        self.wide4 = Wide_Focus(in_channels= 96, out_channels= 96)
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        d44 = d4
        # CAM4
        """ d4 = self.SE0(d4)
        #d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4  """ 
        skip_SA = self.SA(d4)*d4
        skip_CA = self.CA4(d4)*d4
        skip_a = self.AG11(g = skip_SA,x = skip_CA)
        d4 = skip_a
        #d4 = self.ConvBlock4(d4) 
        #d4 = self.stages[3](d4) 
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        #x3 = self.AG3(g=d3,x=skips[0])
        
        # aggregate 3
        #d3 = d3 + x3
        
        # CAM3
        d33 = d3
        """ d3 = self.SE1(d3)
        #d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3    """  
        skip_SA = self.SA(d3)*d3
        skip_CA = self.CA3(d3)*d3
        skip_a = self.AG22(g = skip_SA,x = skip_CA)
        d3 = skip_a
        #d3 = self.ConvBlock3(d3) 
        #d3 = self.stages[2](d3)
        skips3_SA = self.SA(skips[0])*skips[0]
        skips3_CA = self.CA3(skips[0])*skips[0]
        skip3 = self.AG3(g = skips3_SA,x =skips3_CA)
        skip3 = skip3 
        skip3 = self.wide2(skip3)
        """ skip3 = self.CA3(skips[0])*skips[0]
        skip3 = self.SA(skip3)*skip3      
        skip3 = self.ConvBlock3(skip3)  
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3
        d3 = self.stages[2](d3)"""
        d3 = torch.cat([d3,skip3],1)
        d3 = self.concat1(d3) 
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        #x2 = self.AG2(g=d2,x=skips[1])
        
        # aggregate 2
        #d2 = d2 + x2
        
        # CAM2
        
        d22 =d2
        """  d2 = self.SE2(d2)
        #d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2 """
        """ #print(d2.shape) """
        skip_SA = self.SA(d2)*d2
        skip_CA = self.CA2(d2)*d2
        skip_a = self.AG33(g = skip_SA,x = skip_CA)
        d2 = skip_a
        #d2 = self.ConvBlock2(d2) 
        #d2 = self.stages[1](d2)
        """ skip2 = self.CA2(skips[1])*skips[1]
        skip2 = self.SA(skip2)*skip2      
        skip2 = self.ConvBlock2(skip2)  """
        
        """ d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        d2 = self.stages[1](d2) """
        skips2_SA = self.SA(skips[1])*skips[1]
        skip2_CA = self.CA2(skips[1])*skips[1]
        skip2 = self.AG2(g = skips2_SA,x = skip2_CA)
        skip2 = skip2 
        skip2 = self.wide3(skips[1])
        d2 = torch.cat([d2,skip2],1)
        d2 = self.concat2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        #x1 = self.AG1(g=d1,x=skips[2])
        
        # aggregate 1
        #d1 = d1 + x1
        
        # CAM1
        d11 = d1
        """ d1 = self.SE3(d1)
        #d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1 """ 
        skip_SA = self.SA(d1)*d1
        skip_CA = self.CA1(d1)*d1
        skip_a = self.AG44(g = skip_SA,x = skip_CA)
        d1 = self.ConvBlock1(d1) + skip_a
        #d1 = self.stages[0](d1)
        """ short = d1
        d1 = self.stages[0](d1) + short
        skip1 = self.CA1(skips[2])*skips[2]
        skip1 = self.SA(skip1)*skip1      
        skip1 = self.ConvBlock1(skip1) 
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.stages[0](d1) """
        skips1_SA = self.SA(skips[2])*skips[2]
        skip1_CA = self.CA1(skips[2])*skips[2]
        skip1 = self.AG1(g = skips1_SA,x = skip1_CA)
        skip1 = skip1 
        skip1= self.wide4(skips[2])
        d1 = torch.cat([d1,skip1],1)
        d1 = self.concat3(d1)
        return d4, d3, d2, d1
