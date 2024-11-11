import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 对Conv2d层使用Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 批归一化层权重初始化为1
                init.constant_(m.bias, 0)  # 批归一化层偏置初始化为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 全连接层权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 全连接层偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # 定义了一个方法，下面用
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            # x 的维度从[M,32,10]变成[M,32,64]
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        # BN层在通道维度上进行，通常在第2维度上，所以需要交换维度[M,32,64]-->[M,64,32],做批归一化处理
        # 再翻转回来用于后续层处理时维度匹配
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # 第二维为点云数，做最大池化操作，取最能代表pillar的点[M,1,64]
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1) # [M,32,64]
            x_concatenated = torch.cat([x, x_repeat], dim=2) # [M,32,128]
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        # 继承父类初始方法
        super().__init__(model_cfg=model_cfg)
        # 模型配置中提取参数
        self.use_norm = self.model_cfg.USE_NORM  # true 归一化
        self.with_distance = self.model_cfg.WITH_DISTANCE  # false 不考虑距离信息
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ  # true 使用绝对的xyz坐标
        # 点增加6个或者3个特征
        num_point_features += 6 if self.use_absolute_xyz else 3
        #距离特征
        if self.with_distance:
            num_point_features += 1
        # 从模型配置中获取特征编码的卷积层的通道数配置。将输入点云特征的数量(num_point_features)作为第一个通道数，然后依次添加后续的通道数。
        """
        例如，如果num_point_features是3（表示每个点有3个初始特征），
        而self.num_filters是[64, 128]（表示第一层有64个过滤器，第二层有128个过滤器），
        那么num_filters就会是[3, 64, 128]。
        """
        self.num_filters = self.model_cfg.NUM_FILTERS  # [64]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) #[10,64]
        # 创建PFNLayer的列表，PFNLayer是特征编码模块的基本单元。根据通道数配置，构建多个PFNLayer，并添加到列表中。
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            # 当前层输入特征数量,这里第一层为输入层：10个特征向量
            in_filters = num_filters[i]
            # 当前层输出特征数量，这里第二层为卷积层64个卷积核
            out_filters = num_filters[i + 1]
            # 将每一个层遍历添加到PNFLayer
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        # 属性,可以通过索引访问返回列表中的PFNLayer实例
        self.pfn_layers = nn.ModuleList(pfn_layers)
        # 设置体素大小，体素坐标偏移量，将点云坐标转换到体素坐标系
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        # 返回特征编码后的输出特征维度，即最后一个卷积层的通道数。
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        actual_name:(M,)
        max_name:(32,)

        """
        # 拓展维度 (M,1)
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        # 创建了一个与actual_num张量形状相同长度的列表，所有元素初始化为1 (1,1)
        max_num_shape = [1] * len(actual_num.shape)
        # 这行代码将max_num_shape列表中axis + 1位置的元素设置为-1。 (1,-1)
        # 在PyTorch的view方法中，-1表示该维度的大小是自动计算的，以确保整个张量的元素总数保持不变
        max_num_shape[axis + 1] = -1
        # 生成0-体素数目-1的整数序列，并重塑为max_num_shape的形状 (1,32)
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        # [M,32]
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
    batch_dict:
    points：（N,5） -->（batch_index,x,y,z,r）batch_index代表了该点云数据在当前batch中的index
    frame_id：（4，）-->（003877,001908,006616,005355）帧ID
    gt_boxes：（4,40,8）-->（X,Y,Z,dx,dy,dz,ry,CLass）
    use_lead_xyz: (4,) -> (1,1,1,1)
    voxels: (M, 32,4) —> (x,y,z,r)
    voxel_coords：（M,4） -->（batch_index,z,y,x）batch_index代表了该点云数据在当前batch中的index
    voxel_num_points: (M,)
    image_shape：（4,2）每份点云数据对应的2号相机图片分辨率
    batch_size:4
    batch_size大
        """
        # 前向传播函数。接收一个批次的输入数据(batch_dict)，包括体素特征(voxel_features)、体素中点的数量(voxel_num_points)和体素坐标(coords)。
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # [:, :, :3]表示取出每个点的前三个特征，即坐标（x,y,z）,sum(dim=1, keepdim=True)第二个维度（即每个体素内点数量求和），
        # voxel_num_points.type_as(voxel_features).view(-1, 1, 1)将voxel_num_points（一个包含每个体素内点数的一维张量）转换为与voxel_features相同的数据类型。
        # 然后，使用view方法将其形状改变为(-1, 1, 1)，以便能够与前面的坐标和进行广播运算。
        # pillar中所有点的平均坐标points_mean (M,1,3)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        # 并将每个点的坐标减去平均位置得到聚类特征(f_cluster) (M,1,3)
        f_cluster = voxel_features[:, :, :3] - points_mean
        # 计算点云中心特征(f_center)，通过将每个点的x、y、z坐标减去相应的体素偏移量得到。即一个点云到体素中心点的偏移量 (M,1,3)
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        # 根据"use_absolute_xyz"标志位的设置，选择不同的特征组合方式。如果设置为True，将使用原始的点云特征、聚类特征和中心特征作为输入特征。
        # 否则，将只使用点云特征的后3个维度（排除坐标特征），并与聚类特征和中心特征一起作为输入特征。
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        # 沿着每个张量的最后一个维度拼接 [M,32,4+3+3]=(M,32,10)
        features = torch.cat(features, dim=-1)
        # 取出每个体素最大点云数
        voxel_count = features.shape[1]
        """
        由于在生成pillar时，不足32个点云会被0填充，而上面计算会导致由0填充的数据中xc,yc,zc,xp,yp,zp出现值
        所以需要被填充的数据的这些值清0
        """
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0) # (M,32)
        # 在mask最后一个维度添加一个维度，使其与voxel_features匹配，并转换成与voxel_features相同数据类型 (M,32)-->(M,32,1)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        # 将掩码mask应用到特征张量features上。由于mask是一个布尔张量，当与特征张量相乘时
        # ，那些由于填充而对应的掩码值为False（在PyTorch中相当于0）的位置上的特征值会被置为0，
        # 从而实现忽略这些位置上的特征的效果。而那些掩码值为True（相当于1）的位置上的特征值则保持不变。
        features *= mask
        # 通过遍历PFNLayer列表，对输入特征进行多层卷积操作，得到编码后的特征。最后，
        # 将编码后的特征保存在batch_dict中的’pillar_features’键下，并返回更新后的batch_dict。
        for pfn in self.pfn_layers:
            features = pfn(features)
        # squeeze()移除大小为1的维度 这里(M,1,64)-->(M,64),即提取到每个pillar的特征
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
