import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.convnextv2 import *
import torchvision
from lib.danet import PAM_Module
#from lib.convnext import Block


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            #nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            #nn.BatchNorm2d(ch_out),
            #nn.GELU(),
            nn.Dropout(0.3)
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
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int),
            #nn.GELU(),
         
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.GELU()
        
    def forward(self,g,x,kk):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1+x1)
        
        psi = self.psi(psi)

        return kk*psi

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

class MBC(nn.Module):
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
            
        )
        self.layer_dilation2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=2),
            nn.GELU(),
            
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
            
        )
        
        

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer_dilation2(x)
        x3 = self.layer_deepth(x)
        added = x1 + x2 + x3
        x_out = self.layer4(added)
        return added


class Hierarchical_decoder(nn.Module):
    def __init__(self, channels=[512,320,128,64],depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,layer_scale_init_value=1e-6):
        super(Hierarchical_decoder,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
      
        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
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
        
        
        self.conv_mask0 = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.conv_mask1 = nn.Conv2d(channels[1], 1, kernel_size=1)
        self.conv_mask2 = nn.Conv2d(channels[2], 1, kernel_size=1)
        self.conv_mask3 = nn.Conv2d(channels[3], 1, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=2)
        
        self.channel_mul_conv0 = nn.Sequential(
            nn.Conv2d(channels[0], int(channels[0] * 4), kernel_size=1),
            nn.LayerNorm([int(channels[0] * 4), 1, 1]),
            nn.GELU(),  # yapf: disable
            nn.Conv2d(int(channels[0] * 4), channels[0], kernel_size=1))
        
        self.channel_mul_conv1 = nn.Sequential(
            nn.Conv2d(channels[1], int(channels[1] * 4), kernel_size=1),
            nn.LayerNorm([int(channels[1] * 4), 1, 1]),
            nn.GELU(),  # yapf: disable
            nn.Conv2d(int(channels[1] * 4), channels[1], kernel_size=1))
        
        self.channel_mul_conv2 = nn.Sequential(
            nn.Conv2d(channels[2], int(channels[2] * 4), kernel_size=1),
            nn.LayerNorm([int(channels[2] * 4), 1, 1]),
            nn.GELU(),  # yapf: disable
            nn.Conv2d(int(channels[2] * 4), channels[2], kernel_size=1))
        
        self.channel_mul_conv3 = nn.Sequential(
            nn.Conv2d(channels[3], int(channels[3] * 4), kernel_size=1),
            nn.LayerNorm([int(channels[3] * 4), 1, 1]),
            nn.GELU(),  # yapf: disable
            nn.Conv2d(int(channels[3] * 4), channels[3], kernel_size=1))
        
        self.concat1 = conv_block(ch_in=768,ch_out=384)
        self.concat2 = conv_block(ch_in=384,ch_out=192)
        self.concat3 = conv_block(ch_in=192, ch_out=96)

        self.wide1 = MBC(in_channels= 768,out_channels= 768)
        self.wide2 = MBC(in_channels=384, out_channels=384)
        self.wide3 = MBC(in_channels=192, out_channels= 192)
        self.wide4 = MBC(in_channels= 96, out_channels= 96)
        
    def spatial_pool(self, x,i):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
        if(i == 0):
            context_mask = self.conv_mask0(x)
        elif(i == 1):
            context_mask = self.conv_mask1(x)
        elif(i == 2):
            context_mask = self.conv_mask2(x)
        elif(i == 3):
            context_mask = self.conv_mask3(x)
        
            # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        

        return context
        
        
        
    def forward(self,x, skips,i):
        
        
        d4 = self.Conv_1x1(x)
        if(i == 1 ):
            skip_SA = self.CA4(d4)
            skip_CA = self.SA(d4)*d4
            skip_a = self.AG11(g = skip_SA,x = skip_CA,kk = d4)
            d4 = skip_a
        elif(i == 2):
            context = self.spatial_pool(d4,0)
        # [N, C, 1, 1]
            d4 = self.channel_mul_conv0(context)*d4
        
        # upconv3
        d3 = self.Up3(d4)
        
        if(i == 1):
            skip_SA = self.CA3(d3)
            skip_CA = self.SA(d3)*d3
            skip_a = self.AG22(g = skip_SA,x = skip_CA,kk = d3)
            d3 = skip_a
            
        elif(i == 2):
            context = self.spatial_pool(d3,1)
        # [N, C, 1, 1]
            d3 = self.channel_mul_conv1(context) * d3
               
        
        skip3 = self.wide2(skips[0])
        
        d3 = torch.cat([d3,skip3],1)
        d3 = self.concat1(d3) 
        # upconv2
        d2 = self.Up2(d3)
        
        if(i == 1):
            skip_SA = self.CA2(d2)
            skip_CA = self.SA(d2)*d2
            skip_a = self.AG33(g = skip_SA,x = skip_CA,kk = d2)
            d2 = skip_a
        elif(i == 2):
            context = self.spatial_pool(d2,2)
        # [N, C, 1, 1]
            d2 = self.channel_mul_conv2(context) * d2
       
        skip2 = self.wide3(skips[1])
        d2 = torch.cat([d2,skip2],1)
        d2 = self.concat2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        if(i == 1):
            skip_SA = self.CA1(d1)
            skip_CA = self.SA(d1)*d1
            skip_a = self.AG44(g = skip_SA,x = skip_CA,kk=d1)
        elif(i == 2):
            context = self.spatial_pool(d1,3) 
        # [N, C, 1, 1]
            skip_a = self.channel_mul_conv3(context)* d1
            
        d1 = self.ConvBlock1(d1) + skip_a

        skip1= self.wide4(skips[2])
        d1 = torch.cat([d1,skip1],1)
        d1 = self.concat3(d1)
        return d4, d3, d2, d1
