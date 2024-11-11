import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PointAttentionNetwork(nn.Module):
    def __init__(self, C, ratio=8):
        super(PointAttentionNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(C // ratio)
        self.bn2 = nn.BatchNorm1d(C // ratio)
        self.bn3 = nn.BatchNorm1d(C)

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C // ratio, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C // ratio, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, n = x.shape

        a = self.conv1(x).permute(0, 2, 1)  # b, n, c/ratio

        b = self.conv2(x)  # b, c/ratio, n

        s = self.softmax(torch.bmm(a, b))  # b,n,n

        d = self.conv3(x)  # b,c,n
        out = x + torch.bmm(d, s.permute(0, 2, 1))

        return out


class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim=4, output_dim=64):
        super().__init__()
        # 定义线性层，将4维特征映射到32维
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x 的形状为 [M, 4]
        x = self.linear(x)  # 通过线性层
        x = F.relu(x)  # 使用ReLU激活函数
        return x  # 输出形状为 [M, 32]


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000
        # self.PA_test = PointAttentionNetwork(64)

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        # [M,32,64]
        x = F.relu(x)
        # x = self.PA_test(x.permute(0, 2, 1))
        # x = x.permute(0, 2, 1)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        self.PFE1 = MLPFeatureExtractor()
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)  # [10,12,64]

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[2]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.B = self.model_cfg.BIN_NUM
        self.h = self.voxel_z / self.B

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        # 体素内点云平均坐标（M,1,3）
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1)
        points_mean = points_mean.expand(-1, 32, -1)  # (M,32,3)
        # 体素内坐标与体素内点云平均坐标偏移量f_cluster
        f_cluster = voxel_features[:, :, :3] - points_mean  # (M,32,3)

        # 体素中心坐标f_center
        f_center = torch.zeros_like(voxel_features[:, :, :3])  # (M,32,3)
        f_center[:, :, 0] = coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        f_center[:, :, 1] = coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        f_center[:, :, 2] = coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset

        # 体素内坐标与体素中心坐标偏移量f_cluster1
        f_cluster1 = voxel_features[:, :, :3] - f_center
        # f_cluster1[:, :, 0] = voxel_features[:, :, 0] - f_center[:, :, 0]
        # f_cluster1[:, :, 1] = voxel_features[:, :, 1] - f_center[:, :, 1]
        # f_cluster1[:, :, 2] = voxel_features[:, :, 2] - f_center[:, :, 2]

        # 体素内平均点云坐标与全局点云中心坐标偏移量f_cloud
        weighted_sum_points = voxel_features[:, :, :3].sum(dim=1)  # (M, 32, 4)-->(M,3)
        total_weighted_sum = weighted_sum_points.sum(dim=0)  # (3,)
        total_num_points = voxel_num_points.sum()  # 一个标量，总的点数
        overall_point_cloud_center = total_weighted_sum / total_num_points  # (3,)
        overall_point_cloud_center = overall_point_cloud_center.view(1, 1, 3).expand(f_center.shape[0],
                                                                                     f_center.shape[1], 3)  # （M，32，3）
        f_cloud = points_mean - overall_point_cloud_center  # （M,32,3）

        # 每个支柱中心坐标与全局支柱中心偏移量f_pcenter
        center = f_center.sum(dim=1)  # (M,3)
        central_pillar = center.mean(dim=0)  # (3,)
        central_pillar = central_pillar.view(1, 1, 3).expand(f_center.shape[0], f_center.shape[1], 3)  # (M,32,3)
        f_pcenter = f_center - central_pillar  # （M,32,3）

        # 高度编码（直方图）
        P_z = voxel_features[:, :, 2]  # (M,32)
        p_r = voxel_features[:, :, 3]  # (M,32)
        z_index = torch.div(P_z, self.h, rounding_mode='trunc')  # 保持当前行为
        # 确保 z_index 在有效范围内
        z_index = torch.clamp(z_index, 0, self.B - 1)
        # 计算每个体素的 z_index 计数
        # 展平 voxel_features 和 z_index
        z_index_flat = z_index.view(-1).long()
        H_p = torch.bincount(z_index_flat, minlength=self.B)
        H_p = H_p.view(1, 4).expand(voxel_features.shape[0], 4)  # (M,4)

        # H_p = H_p.view(1, 1, 4).expand(voxel_features.shape[0], voxel_features.shape[1], 4)
        # p_r = p_r.view(-1).long()
        # # 创建一个张量用于存储加权反射强度
        # H_r = torch.zeros(self.voxel_z)
        # H_r = H_r.to('cuda:0')
        # # 对每个小体素索引的反射强度进行加权
        # for i in range(self.voxel_z):
        #     indices = (z_index_flat == i)
        #     H_r[i] = torch.sum(p_r[indices])  # (4,)
        # # 归一化加权强度直方图（避免除以 0）
        # H_r = torch.where(H_p > 0, H_r / H_p, H_r)  # (M,32,4)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_cluster1]  # 4+3+3柱内
            features1 = [overall_point_cloud_center, central_pillar, f_cloud, f_pcenter]  # 3+3+3+3柱间
            features2 = [H_p]  # 4高度感知
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        features1 = torch.cat(features1, dim=-1)
        features2 = torch.cat(features2, dim=-1)
        features2 = features2.float()
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        features1 *= mask
        for i in range(len(self.pfn_layers)):
            if i == 0:
                features = self.pfn_layers[i](features)
            else:
                features1 = self.pfn_layers[i](features1)
        # for pfn in self.pfn_layers:
        #     features = pfn(features)
        features = features.squeeze()
        features1 = features1.squeeze()
        features2 = self.PFE1(features2)

        batch_dict['pillar_features'] = features+features1+features2
        return batch_dict