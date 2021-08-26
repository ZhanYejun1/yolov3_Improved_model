from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0   #保证卷积不会对尺寸产生变化
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        # 1*1卷积调整通道数， 3*3卷积进一步提取特征，  减少参数量，使网路更加轻便
        conv2d(in_filters, filters_list[0], 1), # eg.缩小通道数到512
        conv2d(filters_list[0], filters_list[1], 3), # eg.提取特征，通道扩展到1024

        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),

        conv2d(filters_list[1], filters_list[0], 1),  # 输出

        conv2d(filters_list[0], filters_list[1], 3),                       
        # 缩小到75通道数
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True) # 进行分类预测和回归预测
    ])
    return m

class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53(None)

        # out_filters : [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言         
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"]) # 3*（5+20）
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)

        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv2d(512, 256, 1) # 1*1卷积降低通道数
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest') # 上采样，扩大尺寸
        # 26*26*256的特征层
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        52*52*128
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)


    def forward(self, x):
        # 卷积定义时，是放在一起的，现在需要分开
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in #5次卷积的结果保存在out_branch中
            return layer_in, out_branch
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0, out0_branch = _branch(self.last_layer0, x0)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        #x1 = torch.cat([x1,x1], 1)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384

        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2, _ = _branch(self.last_layer2, x2_in)
        return out0, out1, out2

