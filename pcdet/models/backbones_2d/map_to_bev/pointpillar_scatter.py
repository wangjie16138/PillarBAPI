import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size # 空间坐标索引 [432,496,1]
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        步骤：stacked pillar 将生成体素按坐标索引还原到原空间中
        pillar_feature:[M,64]
        coords:[M,4]
        """
        # batch_dict字典中提取体素特征和体素坐标
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        # 创建空列表，用于存储转换为伪图像的数据
        batch_spatial_features = []
        # 获取批次大小，根据坐标索引最大值，获取第一列数值的最大值+1当作是最大索引，第一列的数据是每个点云在当前batch中的索引
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            # 初始化一个零张量
            spatial_feature = torch.zeros(
                self.num_bev_features, # 64
                self.nz * self.nx * self.ny, # 214272
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # 索引coords[:, 0]是否匹配batch_idx来创建掩码，并使用这个掩码来提取当前批次的坐标，即true的坐标。
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :] # (4,3)
            # 伪图像中的索引
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换索引数据类型
            indices = indices.type(torch.long)
            # 与提取坐标类似，这里使用掩码 batch_mask 从 pillar_features 中提取当前批次的体素特征。pillars 现在包含了与 this_coords 对应的体素特征。
            pillars = pillar_features[batch_mask, :] # (4,64)
            # 将 pillars 张量进行转置。这通常是为了调整特征维度以便与空间特征图的形状匹配。
            pillars = pillars.t() #(64,4)
            # 将转置后的体素特征 pillars 放置到 spatial_feature 张量的正确位置上。
            spatial_feature[:, indices] = pillars # 将索引批次的体素特征放入spatial_feature 张量 (64,214272)
            # 将处理完的空间特征 spatial_feature 添加到 batch_spatial_features 列表中
            batch_spatial_features.append(spatial_feature)
        #用 torch.stack 函数将 batch_spatial_features 列表中的所有空间特征张量沿着第一个维度（即批次维度）堆叠起来。
        batch_spatial_features = torch.stack(batch_spatial_features, 0) #(4,64,214272)
        # view 方法来改变 batch_spatial_features 张量的形状。batch_size 是批次中样本的数量，self.num_bev_features 是每个体素（pillar）的特征数量，
        # self.nz、self.ny 和 self.nx 分别是z、y和x方向上的体素数量
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx) # (4,64,496,432)
        #将重塑后的空间特征张量 batch_spatial_features 添加到 batch_dict 字典中，键为 'spatial_features'。
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict