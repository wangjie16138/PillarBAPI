import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    """
        BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [3, 5, 5]
        LAYER_STRIDES: [2, 2, 2]
        NUM_FILTERS: [64, 128, 256]
        UPSAMPLE_STRIDES: [1, 2, 4]
        NUM_UPSAMPLE_FILTERS: [128, 128, 128]
    """
    def __init__(self, model_cfg, input_channels):
        # 定义了一个名为 BaseBEVBackbone 的类，继承自 nn.Module。__init__ 方法用于初始化模型，接收模型配置 model_cfg 和输入通道数 input_channels
        super().__init__()
        self.model_cfg = model_cfg
        # 检查了模型配置中是否有关于卷积层数量 (LAYER_NUMS)、步幅 (LAYER_STRIDES) 和滤波器数量 (NUM_FILTERS) 的定义。
        # 如果这些配置存在，就获取它们的值并确保它们的长度一致；否则，将这些参数设为空列表。
        # 读取下采样参数
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []
        # 检查模型配置中是否有定义关于上采样步幅 (UPSAMPLE_STRIDES) 和上采样滤波器数量 (NUM_UPSAMPLE_FILTERS) 的参数，
        # 如果有则获取其值并确保长度一致；否则将这些参数设为空列表。
        # 读取上采样参数
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        # 计算了卷积层数量，并初始化了一个 c_in_list 列表，其中包含输入通道数和各个卷积层的滤波器数量（除了最后一层）。然后，创建了 blocks 和 deblocks 两个空的 nn.ModuleList()，用于存储卷积块和反卷积块。
        # 在 for 循环中，对于每个卷积层（level），生成了一系列卷积相关的层次结构，包括零填充 (ZeroPad2d)、卷积 (Conv2d)、批归一化 (BatchNorm2d) 和 ReLU 激活函数。
        num_levels = len(layer_nums) # 3
        c_in_list = [input_channels, *num_filters[:-1]] # [64,64,128]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3, # (64,64)-->(64,128)-->(128,256)这一层只是遍历卷积层
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            # 对于每个卷积层的每个卷积操作，都使用了相同的卷积核大小和填充，但在中间层次结构上重复了多次卷积、批归一化和 ReLU 操作。
            # 最后，使用 nn.Sequential 将这些层次结构组合成一个整体的卷积块，并将其添加到 blocks 中。
            # 这段代码主要负责定义了卷积块的结构，用于在模型中提取特征。
            for k in range(layer_nums[idx]): # 定义卷积块结构，基于卷积层[3,5,5]遍历卷积块,堆叠卷积层，注意卷积核输入输入通道数一样
                # 循环三批[[(64,64),(64,64),(64,64)],[(128,128),(128,128),(128,128),(128,128),(128,128)],[(256,256),(2256,256),(256,256),(256,256),(256,256)]]
                # 每个括号就是一个卷积核
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0: # (1,2,4)
                stride = upsample_strides[idx] # 1,2,4
                # 判断使用哪种卷积，这里使用转置卷积，即反卷积
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features (4,64,496,432)
                block:[[3,3,5],[[(64,64),(64,64),(64,64)],[(128,128),(128,128),(128,128),(128,128),(128,128)],[(256,256),(256,256),(256,256),(256,256),(256,256)]]]
        Returns:
        """
        spatial_features = data_dict['spatial_features'] # (4,64,496,432)
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)): # 3 遍历三个卷积层 (4,64,496,432)-->(4,64,248,216)-->(4,128,124,108)-->(4,256,62,54)
            x = self.blocks[i](x)
            # 计算步长，原始特征图宽/通过卷积层特征图宽
            stride = int(spatial_features.shape[2] / x.shape[2])
            # 保存数据
            ret_dict['spatial_features_%dx' % stride] = x
            # 判断是否存在上采样模块 有就将每阶段的处理的特征图通过上采样模块，这里是反卷积层，存入ups
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        """
        经过所有上采样层得到3个尺度信息
        每个尺度的shape都是(batch_size,128,248,216)
        这里len(ups)>1判断为真，沿着每个尺度的第一个维度拼接，得到(batch_size,384,248,246)
        """
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        # 上采样层多，则需要补充特征变换
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


class BaseBEVResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
