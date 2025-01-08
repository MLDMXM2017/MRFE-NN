import torch.nn as nn
#import torch
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
# 重写conv2d
import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
# 实现tensorflow的padding=same，自动计算填充边缘数
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)
class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd): 

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)   
'''
# 池化输出相同尺寸
#input_size: 输入尺寸 output_size: 输出尺寸
stride = intput_size // output_size #步长 为1
kernel_size = input_size - ( output_size -1 ) * stride # 核的尺寸为1
padding = 0 
'''                        
class SoftPool2D(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SoftPool2D,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

    def soft_pool2d(self, x, kernel_size=2, stride=None):
        kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = (stride, stride)
        _, c, h, w = x.shape
        e_x = torch.sum(torch.exp(x),axis=1,keepdim=True)
        return F.avg_pool2d(x * e_x, kernel_size, stride=stride) * (sum(kernel_size))/(F.avg_pool2d(e_x, kernel_size, stride=stride) * (sum(kernel_size)))


class BottleneckLayer(nn.Module):
    def __init__(self, channels_in, growth_rate):
        super().__init__()
        self.growth_rate = growth_rate
        self.channels_in = channels_in
        #self.out_channels_1x1 = 4*self.growth_rate # 4为bottleneck_width
        #self.out_channels_1x1 = self.growth_rate
        self.layers = nn.Sequential(nn.BatchNorm2d(num_features=self.channels_in),# 若不需要bottleneck
                                    nn.ReLU(),
                                    #Conv2d(in_channels=self.channels_in, out_channels=self.growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
                                    Conv2d(in_channels=self.channels_in, out_channels=self.channels_in, kernel_size=3, stride=1, padding=self.growth_rate, dilation=self.growth_rate,bias=False),
                                    nn.Dropout(0.1))
    def forward(self, x):
        out = self.layers(x)
        # 重点：这里是x前面所有层的输出特征图
        #out = torch.cat((x, out), dim=1)
        return out
 
class TransitionLayer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        # 1：定义转换层的输入输出通道数
        self.channels_in = channels_in
        self.channels_out = channels_out
        # 2：BN+ReLU+Conv1x1+AvgPool2x2
        self.layers = nn.Sequential(nn.BatchNorm2d(num_features=channels_in),
                                    nn.ReLU(),
                                    Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=1, stride=1, padding=1, bias=False),
                                    nn.Dropout(0.1),
                                    nn.AvgPool2d(kernel_size=2))

    def forward(self, x):
        out = self.layers(x)
        return out

def make_dense_block(num_bottleneck, growth_rate, channels_in):
    """
    根据Bottleneck制作Dense Block
    :param num_bottleneck: 目标Dense Block层数
    :param growth_rate: 增长率，即通道数
    :param channels_in: 输入通道数
    :return: 返回nn.Sequential类型的Dense Block
    """
    # 1：创建容器
    layers = []
    # 2：每一个bottleneck层的输入通道数是前面所有bottleneck层输出通道数之和
    #    每一个bottleneck层输出通道数都是增长率k，即论文中growth rate
    current_channels = channels_in
    for i in range(num_bottleneck):
        # 3：给Dense Block添加Bottleneck层
        layers.append(BottleneckLayer(channels_in=current_channels, growth_rate=growth_rate))
        # 4：每次添加current_channels都增大growth rate
        #current_channels += growth_rate
    return nn.Sequential(*layers)

class FirstConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        """
        DenseNet第一个卷积层，将输入图片从3通道变为其它自定义通道数
        :param channels_in: 输入图片通道数
        :param channels_out: 自己设定的输出通道数
        """
        super().__init__()
        self.layers = nn.Sequential(Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, stride=1, padding=1, bias=False)
                                    ,nn.BatchNorm2d(num_features=channels_out)
                                    ,nn.ReLU()
                                    #,nn.AvgPool2d(kernel_size=2)
                                    )
    def forward(self, x):
        return self.layers(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
        
class MaxCoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(MaxCoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        #self.pool_h = SoftPool2D(1,1)
        #self.pool_w = SoftPool2D(1,1)


        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size() # （64，1，44，44）
        x_h = self.pool_h(x) # （64，1，44，1）
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # （64，1，1，44）→（64，1，44，1）

        y = torch.cat([x_h, x_w], dim=2) #（64，1，88，1）
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2) #（64，1，88，1）→ （64，1，44，1），（64，1，44，1）
        x_w = x_w.permute(0, 1, 3, 2) # （64，1，44，1）→（64，1，1，44）

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
        
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size() # （64，1，44，44）
        
        x_h = self.pool_h(x) # （64，1，44，1）
        x_h_max = self.max_pool_h(x) # （64，1，44，1）
        
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # （64，1，1，44）→（64，1，44，1）
        x_w_max = self.max_pool_w(x).permute(0, 1, 3, 2) # （64，1，1，44）→（64，1，44，1）
        
        x_h = torch.cat([x_h, x_h_max], dim=2) #（64，1，88，1）
        x_w = torch.cat([x_w, x_w_max], dim=2) #（64，1，88，1）
  
        y = torch.cat([x_h, x_w], dim=2) #（64，1，88*2，1）
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h*2, w*2], dim=2) #（64，1，88*2，1）→ （64，1，88，1），（64，1，88，1）
        x_w = x_w.permute(0, 1, 3, 2) # （64，1，88，1）→（64，1，1，88）

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

             
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        
'''
class SpatialAvgAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAvgAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(avg_out)
        return self.sigmoid(x)
        
class SpatialMaxAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialMaxAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(max_out)
        return self.sigmoid(x)
'''                
        
def attention_enhance(x):
    # 空间注意力区域增强，其它区域元素置为0
    # 样本数
    num_rois = x.shape[0]  
    # 通道数，如24
    num_channel = x.shape[1]
    # 特征图尺寸，如44
    H = x.shape[2]
    # 压成 HW = 1936
    HW = x.shape[2] * x.shape[3]
    # 注意力区域（保留前50%）
    spatial_drop_num = math.ceil(HW * 1 / 2.0) 
    
    #x = x.clone().detach()
    # [-1,24,44,44] -> [-1,24,1936]   对压平的1936个特征梯度求均值channel_mean  [-1,24,1] => view [-1,24,1,1] grad_channel_mean
    grad_channel_mean = torch.mean(x.view(num_rois, num_channel, -1), dim=2)
    grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
    # [-1,24,44,44] -> dim=1求和，24个通道上的特征求和 [-1,1,44,44]
    spatial_mean = torch.sum(x * grad_channel_mean, 1)
    # [-1,1,44,44] => view [-1,1936]
    spatial_mean = spatial_mean.view(num_rois, HW)
    
    # 返回大于等于参数x的最小整数，例如HW=121  121/3，返回41
    # spatial_mean [-1,1936] -1表示样本数，对num_rois个样本按1936个梯度值降序排序，取第spatial_drop_num+1个梯度值作为mask阈值
    th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
    # 构造一个阈值矩阵 th18_mask_value
    th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW) 
    #构造mask矩阵，spatial_mean的特征值若小于th18_mask_value，则取值0，否则取值1，例如（以num_rois = 2， HW = 49为例）：
    
    mask_all_cuda = torch.where(spatial_mean < th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                torch.ones(spatial_mean.shape).cuda())
    # （-1,1936）-> reshape+view (-1,1,44,44)
    mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
    
    x_1 = x * mask_all  # x_new:[-1,384,11,11]  mask_all:(-1,1,11,11) or (-1,384,1,1)     
    return x_1
       
class myNet(nn.Module):
    def __init__(self, growth_rate, channels_in, num_dense_block, num_bottleneck, num_channels_before_dense, compression, num_classes):
        """
        DenseNet核心代码
        :param growth_rate: 增长率
        :param channels_in: 输入数据通道数
        :param num_dense_block: 需要几个Dense Block，暂时不用此参数
        :param num_bottleneck: 用list表示每个DenseBlock包含的bottleneck个数，如list(6, 12, 24, 16)表示DenseNet121
        :param num_channels_before_dense: 第一个卷积层的输出通道数
        :param compression: 压缩率，Transition层的输出通道数为Compression乘输入通道数
        :param num_classes:类别数
        """
        super().__init__()
        self.growth_rate = growth_rate
        self.channel_in = channels_in
        self.num_dense_block = num_dense_block
        self.num_bottleneck = num_bottleneck

        # 1：定义第1个卷积层
        self.first_conv = FirstConv(channels_in=channels_in, channels_out=num_channels_before_dense)
        
        dense_1_out_channels = num_channels_before_dense
        dense_2_out_channels = dense_1_out_channels * 2
        dense_3_out_channels = dense_2_out_channels * 2
        
        # 2：定义第1个Dense Block，其输出通道数为输入通道数加上层数*增长率
        self.dense_1 = make_dense_block(num_bottleneck=num_bottleneck[0], channels_in=num_channels_before_dense,
                                        growth_rate=growth_rate)
        self.dense_1_1 = make_dense_block(num_bottleneck=num_bottleneck[0], channels_in=num_channels_before_dense,
                                        growth_rate=growth_rate+1)
        #dense_1_out_channels = int(num_channels_before_dense + num_bottleneck[0]*growth_rate)
        # 注意力机制
        #self.ca_1 = ChannelAttention(dense_1_out_channels,12)
        self.spa_1 = SpatialAttention()
        #self.spa_1 = MaxCoordAtt(dense_1_out_channels,dense_1_out_channels)
        self.coord_1 = MaxCoordAtt(dense_1_out_channels,dense_1_out_channels)
        #self.spa_max_1 = SpatialMaxAttention()
        #self.nonlocal_1 = NonLocalBlockND(in_channels=dense_1_out_channels)
        
        self.transition_1 = TransitionLayer(channels_in=dense_1_out_channels*2,
                                            channels_out=int(compression*dense_2_out_channels))

        # 3：定义第2个Dense Block，其输出通道数为输入通道数加上层数*增长率
        self.dense_2 = make_dense_block(num_bottleneck=num_bottleneck[1], channels_in=int(compression*dense_2_out_channels),
                                        growth_rate=growth_rate)
        self.dense_2_2 = make_dense_block(num_bottleneck=num_bottleneck[1], channels_in=int(compression*dense_2_out_channels),
                                        growth_rate=growth_rate+1)
        #dense_2_out_channels = int(compression*dense_1_out_channels + num_bottleneck[1]*growth_rate)
        # 注意力机制
        #self.ca_2 = ChannelAttention(dense_2_out_channels,12)
        #self.spa_avg_2 = SpatialAvgAttention()
        #self.spa_max_2 = SpatialMaxAttention()
        self.spa_2 = SpatialAttention()
        #self.spa_2 = MaxCoordAtt(dense_2_out_channels,dense_2_out_channels)
        self.coord_2 = MaxCoordAtt(dense_2_out_channels,dense_2_out_channels)
        #self.nonlocal_2 = NonLocalBlockND(in_channels=dense_2_out_channels)
        
        self.transition_2 = TransitionLayer(channels_in=dense_2_out_channels*2,
                                            channels_out=int(compression*dense_3_out_channels))

        # 4：定义第3个Dense Block，其输出通道数为输入通道数加上层数*增长率
        self.dense_3 = make_dense_block(num_bottleneck=num_bottleneck[2], channels_in=int(compression * dense_3_out_channels),
                                        growth_rate=growth_rate)
        self.dense_3_3 = make_dense_block(num_bottleneck=num_bottleneck[2], channels_in=int(compression * dense_3_out_channels),
                                        growth_rate=growth_rate+1)
        #dense_3_out_channels = int(compression * dense_2_out_channels + num_bottleneck[2] * growth_rate)
       # self.transition_3 = TransitionLayer(channels_in=dense_3_out_channels,
                                            #channels_out=int(compression * dense_3_out_channels))

        # 5：定义第4个Dense Block，其输出通道数为输入通道数加上层数 * 增长率
        #self.dense_4 = make_dense_block(num_bottleneck=num_bottleneck[3],
         #                               channels_in=int(compression * dense_3_out_channels),
          #                              growth_rate=growth_rate)
        #dense_4_out_channels = int(compression * dense_3_out_channels + num_bottleneck[3] * growth_rate)

        # 6：定义最后的7x7池化层，和分类全连接层
        #self.BN_before_classify = nn.BatchNorm2d(num_features=dense_4_out_channels)
        self.BN_before_classify = nn.BatchNorm2d(num_features=dense_3_out_channels)
        #self.pool_before_classify = nn.AvgPool2d(kernel_size=7,stride=1)
        #self.classify = nn.Linear(in_features=dense_4_out_channels, out_features=num_classes)
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.classify = nn.Linear(in_features=dense_3_out_channels, out_features=num_classes)


    def forward(self, x):
        x = self.first_conv(x)
        
        x1 = self.dense_1(x)
        x1 = self.spa_1(x1) * x1
        
        x2 = self.dense_1_1(x)
        x2 = self.coord_1(x2)
        
        x1_enhance = attention_enhance(x1)
        x2_enhance = attention_enhance(x2)

        '''
        # 可视化 mean 和 max 的图像（检验质量）
        plt.figure(figsize=(10,20))
        plt.subplot(1,3,1)
        plt.imshow(x.detach().cpu().numpy()[0,0,:,:],cmap='binary')    
        plt.subplot(1,3,2)
        plt.imshow(x1_enhance.detach().cpu().numpy()[0,0,:,:],cmap='binary')      
        plt.subplot(1,3,3)   
        plt.imshow(x2_enhance.detach().cpu().numpy()[0,0,:,:],cmap='binary')     
        plt.savefig('max_spa_coord_1_enhance.png', bbox_inches= 'tight', pad_inches= 0)  
        '''
        
        # 拼接
        x1_enhance = torch.cat((x1_enhance, x1), dim=1)
        #print('x_1:',x_1.shape)
        x2_enhance = torch.cat((x2_enhance, x2), dim=1)
        '''
        # 两路的特征图叠加 
        lam = 0.7
        x1_enhance = lam * x1_enhance + (1-lam) * x2_enhance
        x2_enhance = (1-lam) * x1_enhance + lam * x2_enhance
        
        # 随机交换两路的特征图（交换的比例50%） 
        # 先把特征图按通道分成4组，再打乱顺序
        N, C, H, W = x1_enhance.size()
      
        groups = 4
        N, C, H, W = x1_enhance.size()
        x1_enhance = x1_enhance.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
        x2_enhance = x2_enhance.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
        
        x1_enhance_temp = x1_enhance.permute(1, 0, 2, 3) # n,c,h,w   → c, n, h, w
        x2_enhance_temp = x2_enhance.permute(1, 0, 2, 3) # n,c,h,w   → c, n, h, w 

        # 将第一路、第二路特征的前半通道互相交换
        #a = x1_enhance_temp[0:C//4,:,:,:]
        #b = x2_enhance_temp[0:C//4,:,:,:]
        x1_enhance_temp[0:C//4,:,:,:] = x2_enhance_temp[0:C//4,:,:,:]
        x2_enhance_temp[0:C//4,:,:,:] = x1_enhance_temp[0:C//4,:,:,:]
        x1_enhance = x1_enhance_temp.permute(1, 0, 2, 3) # c, n, h, w  → n,c,h,w 
        x2_enhance = x2_enhance_temp.permute(1, 0, 2, 3) # c, n, h, w  → n,c,h,w 
        '''
                
        x1_enhance = self.transition_1(x1_enhance)
        x1_enhance = self.dense_2(x1_enhance)
        x2_enhance = self.transition_1(x2_enhance)
        x2_enhance = self.dense_2_2(x2_enhance)
        #x = self.ca_2(x) * x

        x1_enhance = self.spa_2(x1_enhance) * x1_enhance
        #x2_enhance = self.nonlocal_2(x2_enhance)
        #x1_enhance = self.spa_avg_2(x1_enhance) * x1_enhance
        x2_enhance = self.coord_2(x2_enhance) 
        
        #print('sa_2:',x.shape) (-1,24,22,22)
        #print('sa_2:',x)
        
        x1_enhance_2 = attention_enhance(x1_enhance)
        x2_enhance_2 = attention_enhance(x2_enhance)
        
        '''
        # 可视化 mean 和 max 的图像（检验质量）
        plt.figure(figsize=(10,20))
        plt.subplot(1,3,1)
        plt.imshow(x1_enhance.detach().cpu().numpy()[0,0,:,:],cmap='binary')  
        plt.subplot(1,3,2)
        plt.imshow(x1_enhance_2.detach().cpu().numpy()[0,0,:,:],cmap='binary')      
        plt.subplot(1,3,3)   
        plt.imshow(x2_enhance_2.detach().cpu().numpy()[0,0,:,:],cmap='binary')     
        plt.savefig('max_spa_coord_2_enhance.png', bbox_inches= 'tight', pad_inches= 0)  
        '''
          
        # 拼接
        x1_enhance_2 = torch.cat((x1_enhance_2, x1_enhance), dim=1)
        x2_enhance_2 = torch.cat((x2_enhance_2, x2_enhance), dim=1)
 
        
        x1_enhance_2 = self.transition_2(x1_enhance_2)
        x1_enhance_2 = self.dense_3(x1_enhance_2)
        x1_enhance_2 = self.BN_before_classify(x1_enhance_2)
        
        x2_enhance_2 = self.transition_2(x2_enhance_2)
        x2_enhance_2 = self.dense_3_3(x2_enhance_2)
        x2_enhance_2 = self.BN_before_classify(x2_enhance_2)
        
        x1_enhance_2 = F.adaptive_avg_pool2d(x1_enhance_2, (1,1))
        x1_enhance_2_dr1 = self.dropout_1(x1_enhance_2.view(x1_enhance_2.size(0), -1))
        x1_enhance_2_dr2 = self.dropout_2(x1_enhance_2.view(x1_enhance_2.size(0), -1))
        
        x1_enhance_2 = self.classify(x1_enhance_2.view(x1_enhance_2.size(0), -1))
        x1_enhance_2_dr1 = self.classify(x1_enhance_2_dr1)
        x1_enhance_2_dr2 = self.classify(x1_enhance_2_dr2)
        
        x2_enhance_2 = F.adaptive_avg_pool2d(x2_enhance_2, (1,1))
        x2_enhance_2_dr1 = self.dropout_1(x2_enhance_2.view(x2_enhance_2.size(0), -1))
        x2_enhance_2_dr2 = self.dropout_2(x2_enhance_2.view(x2_enhance_2.size(0), -1))
        
        x2_enhance_2 = self.classify(x2_enhance_2.view(x2_enhance_2.size(0), -1))
        x2_enhance_2_dr1 = self.classify(x2_enhance_2_dr1)
        x2_enhance_2_dr2 = self.classify(x2_enhance_2_dr2)
        
        way1_logits1 = x1_enhance_2.reshape(1,-1,7)
        way1_logits2 = x1_enhance_2_dr1.reshape(1,-1,7)
        way1_logits3 = x1_enhance_2_dr2.reshape(1,-1,7)
        
        way1_out = torch.cat((way1_logits1, way1_logits2, way1_logits3), dim=0) 
        way1_mean_logits = torch.mean(way1_out, dim=0, keepdim=False)
        way1_var = torch.var(way1_out, dim=0, keepdim=False) 
        way1_var_sum = torch.sum(way1_var, dim=1, keepdim=False)
          
        way2_logits1 = x2_enhance_2.reshape(1,-1,7)   
        way2_logits2 = x2_enhance_2_dr1.reshape(1,-1,7)
        way2_logits3 = x2_enhance_2_dr2.reshape(1,-1,7)
        
        way2_out = torch.cat((way2_logits1, way2_logits2, way2_logits3), dim=0) 
        way2_mean_logits = torch.mean(way2_out, dim=0, keepdim=False)    
        way2_var = torch.var(way2_out, dim=0, keepdim=False)
        way2_var_sum = torch.sum(way2_var, dim=1, keepdim=False)
        
        way1_mean_var = torch.mean(way1_var_sum, dim=0, keepdim=False)   
        way2_mean_var = torch.mean(way2_var_sum, dim=0, keepdim=False) 
        #print("way1_mean_var:", way1_mean_var)   
        #print("way2_mean_var:", way2_mean_var)  
        exp_sum = math.exp(way1_mean_var) + math.exp(way2_mean_var) 
        w1 = math.exp(way2_mean_var) / exp_sum
        w2 = math.exp(way1_mean_var) / exp_sum
        #print("w1:",w1)
        #print("w2:",w2)
        
        mean_out = w1 * way1_mean_logits + w2 * way2_mean_logits
        #print("way1_mean_logits[0]:",way1_mean_logits[0])
        #print("way2_mean_logits[0]:",way2_mean_logits[0])
        #print("mean_out[0]:",mean_out[0])
        
        return mean_out

#x = torch.randn(size=(4, 1, 44, 44))
#densenet = DenseNet(channels_in=1, compression=0.5, growth_rate=12, num_classes=7,num_bottleneck=[18, 18, 18],
 #                       num_channels_before_dense=32,
  #                      num_dense_block=3)
#out = densenet(x)
#print(densenet)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
#model = DenseNet(channels_in=1, compression=0.5, growth_rate=12, num_classes=7,num_bottleneck=[18, 18, 18],
#                        num_channels_before_dense=24,
#                        num_dense_block=3).to(device)

#summary(model, (1, 44, 44))