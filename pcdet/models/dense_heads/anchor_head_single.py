import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        #使用 super() 调用父类的初始化方法，并传递一系列参数来配置头部网络
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        #计算了每个位置的锚点数量之和
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        #创建了一个卷积层 self.conv_cls，用于预测每个锚点位置的类别概率。
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        #创建了另一个卷积层 self.conv_box，用于预测每个锚点位置的边界框坐标。
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        #根据配置选择是否创建方向分类器的卷积层
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        #初始化网络层的权重
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        #self.conv_cls.bias 的初始化使用了常数初始化方法，根据逻辑来设置偏置的初始值。
        #self.conv_box.weight 使用了正态分布初始化方法，给权重设置了均值和标准差。
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        #执行前向传播，提取了空间特征 spatial_features_2d。
        spatial_features_2d = data_dict['spatial_features_2d']
        #特征通过 self.conv_cls 和 self.conv_box 卷积层进行处理，生成类别预测 cls_preds 和边界框预测 box_preds。
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        #调整预测结果的形状
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        #如果启用了方向分类器，将空间特征通过 self.conv_dir_cls 卷积层处理，生成方向预测 dir_cls_preds
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        #如果处于训练状态，调用 self.assign_targets 方法为每个锚点位置分配目标，并将结果更新到 self.forward_ret_dict 中
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
        #生成预测的框
        #如果不处于训练状态，或者在训练时需要预测框 (predict_boxes_when_training=True)，则执行以下操作：
        if not self.training or self.predict_boxes_when_training:
            # 调用 self.generate_predicted_boxes 方法生成预测的类别和边界框。
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            # 将生成的类别预测 batch_cls_preds和边界框预测batch_box_preds存储到 data_dict 中。
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            # 设置 cls_preds_normalized 标志为 False，表示类别预测未经归一化。
            data_dict['cls_preds_normalized'] = False

        return data_dict
