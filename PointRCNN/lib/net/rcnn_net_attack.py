import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
from lib.rpn.proposal_target_layer import ProposalTargetLayer
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import numpy as np

class RCNNNet(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]

            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=cfg.RCNN.SA_CONFIG.RADIUS[k],
                    nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RCNN.USE_BN
                )
            )
            channel_in = mlps[-1]

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel = reg_channel + (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)
                
                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)

                target_dict['pts_input'] = pts_input
            else:
                # with torch.no_grad():
                #print('-------------',input_data.keys())
                target_dict = self.get_input(input_data)

                pts_input = target_dict['pooled_features'].view(-1, target_dict['pooled_features'].shape[2], target_dict['pooled_features'].shape[3])
                target_dict['pts_input'] = pts_input

                # with torch.no_grad():
                    # target_dict = self.proposal_target_layer(input_data)
                # pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                # target_dict['pts_input'] = pts_input
        else:
            pts_input = input_data['pts_input']
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input']
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d']
            if self.training:
                target_dict['cls_label'] = input_data['cls_label']
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask']
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct']

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].clone().transpose(1, 2).unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].clone().transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]

        else:
            l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        rcnn_cls = self.cls_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        # if self.training:
        ret_dict.update(target_dict)
        return ret_dict

    def get_input(self, input_data):
        
        rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']

        ### add by wen
        roi_boxes3d, gt_boxes3d = input_data['roi_boxes3d'], input_data['gt_boxes3d']
        batch_rois, batch_gt_of_rois, batch_roi_iou = self.get_gt_rois(roi_boxes3d, gt_boxes3d)


        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                    input_data['seg_mask'].unsqueeze(dim=2)]
        else:
            pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

        if cfg.RCNN.USE_DEPTH:
            pts_depth = input_data['pts_depth'] / 70.0 - 0.5
            pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
        pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

        pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)

        pooled_features, pooled_empty_flag = \
                roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                              sampled_pt_num=cfg.RCNN.NUM_POINTS)

        # canonical transformation
        batch_size = batch_rois.shape[0]
        roi_ry = batch_rois[:, :, 6] % (2 * np.pi)   # added by wen
        roi_center = batch_rois[:, :, 0:3]
        batch_gt_of_rois[:, :, 0:3] = batch_gt_of_rois[:, :, 0:3] - roi_center  # added by wen
        batch_gt_of_rois[:, :, 6] = batch_gt_of_rois[:, :, 6] - roi_ry  # added by wen
        pooled_features[:, :, :, 0:3] = pooled_features[:, :, :, 0:3].clone() - roi_center.unsqueeze(dim=2)
        for k in range(batch_size):
            pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3].clone(),
                                                                                batch_rois[k, :, 6])
            batch_gt_of_rois[k] = kitti_utils.rotate_pc_along_y_torch(batch_gt_of_rois[k].unsqueeze(dim=1),
                                                  roi_ry[k]).squeeze(dim=1)  # added by wen



        # regression valid mask
        valid_mask = (pooled_empty_flag == 0)
        reg_valid_mask = ((batch_roi_iou > cfg.RCNN.REG_FG_THRESH) & valid_mask).long()
 
        # classification label
        batch_cls_label = (batch_roi_iou > cfg.RCNN.CLS_FG_THRESH).long()
        invalid_mask = (batch_roi_iou > cfg.RCNN.CLS_BG_THRESH) & (batch_roi_iou < cfg.RCNN.CLS_FG_THRESH)
        batch_cls_label[valid_mask == 0] = -1
        batch_cls_label[invalid_mask > 0] = -1

        # torch.Size([4, 64])
        #print(pooled_features.shape)

        output_dict = {'cls_label': batch_cls_label.view(-1),
        'reg_valid_mask': reg_valid_mask.view(-1),
        'gt_of_rois': batch_gt_of_rois.view(-1, 7),
        'gt_iou': batch_roi_iou.view(-1),
        'roi_boxes3d': batch_rois.view(-1, 7),
        'pooled_features': pooled_features
        }

        return output_dict

    def get_gt_rois(self, roi_boxes3d, gt_boxes3d):
        """
        :param roi_boxes3d: (B, M, 7)
        :param gt_boxes3d: (B, N, 8) [x, y, z, h, w, l, ry, cls]
        :return
            batch_rois: (B, N, 7)
            batch_gt_of_rois: (B, N, 8)
            batch_roi_iou: (B, N)
        """

        batch_size = roi_boxes3d.size(0)
        N = roi_boxes3d.size(1)
        # fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))
        # print('batch_size, fg_rois_per_image, cfg.RCNN.FG_RATIO, cfg.RCNN.ROI_PER_IMAGE', batch_size, fg_rois_per_image, cfg.RCNN.FG_RATIO, cfg.RCNN.ROI_PER_IMAGE)
        # batch_size, fg_rois_per_image, cfg.RCNN.FG_RATIO, cfg.RCNN.ROI_PER_IMAGE 4 32 0.5 64
        # batch_rois = torgt_boxes3d.new(batch_size, N, 7).zero_()
        # batch_gt_of_rois = gt_boxes3d.new(batch_size, N, 7).zero_()
        # batch_roi_iou = gt_boxes3d.new(batch_size, N).zero_()

        batch_rois = torch.zeros((batch_size, N, 7)).to(self.device)
        batch_gt_of_rois = torch.zeros((batch_size, N, 7)).to(self.device)
        batch_roi_iou = torch.zeros((batch_size, N)).to(self.device)


        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]
            # print('cur_roi, cur_gt', cur_roi.shape, cur_gt.shape)

            if(len(cur_gt)==0):
                cur_gt=torch.from_numpy(np.zeros([1,7])).cuda(non_blocking=True).float()
            # cur_roi, cur_gt torch.Size([100, 7]) torch.Size([1, 7])
            # print()
            # k = len(cur_gt) - 1
            # print('k', k)
            # while cur_gt[k].sum() == 0:
            #     # print(cur_gt[k])
            #     k = k - 1
            # print('k', k)
            # cur_gt_valid = cur_gt[:k + 1]
            # print('cur_gt', cur_gt.shape)
            # cur_gt torch.Size([1, 7])
            # include gt boxes in the candidate rois
            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
            # print(iou3d)
            # print('iou3d', iou3d.shape)
            # iou3d torch.Size([100, 1])
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            # print(max_overlaps, gt_assignment)
            # print('max_overlaps, gt_assignment', max_overlaps.shape, gt_assignment.shape)
            # max_overlaps, gt_assignment torch.Size([100]) torch.Size([100])

            batch_rois[idx] = cur_roi
            batch_gt_of_rois[idx] = cur_gt[gt_assignment]
            batch_roi_iou[idx] = max_overlaps

            # print(batch_rois[idx].shape, batch_gt_of_rois[idx].shape, batch_roi_iou[idx].shape)
            # torch.Size([100, 7]) torch.Size([100, 7]) torch.Size([100])

        return batch_rois, batch_gt_of_rois, batch_roi_iou
