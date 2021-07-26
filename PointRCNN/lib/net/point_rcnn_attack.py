import torch
import torch.nn as nn
from lib.net.rpn_attack import RPN
from lib.net.rcnn_net import RCNNNet
from lib.net.rcnn_net_attack import RCNNNet as RCNNNet_Attack
from lib.config import cfg

class AttackPointRCNN_RPN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TEST'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            #with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
            if cfg.RPN.FIXED:
                self.rpn.eval()
            rpn_output = self.rpn(input_data)
            output.update(rpn_output)

            # rcnn inference
            if cfg.RCNN.ENABLED:

                # self.rcnn_net.eval()

                rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                rpn_scores_raw = rpn_cls[:, :, 0]
                rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                # proposal layer
                rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                output['rois'] = rois
                output['roi_scores_raw'] = roi_scores_raw
                output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output


class AttackPointRCNN_RCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TEST'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet_Attack(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            #with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
            if cfg.RPN.FIXED:
                self.rpn.eval()
            rpn_output = self.rpn(input_data)
            output.update(rpn_output)

            # rcnn inference
            if cfg.RCNN.ENABLED:

                rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                # print(rpn_cls)
                # print('---grad')
                # print(rpn_cls.requires_grad, rpn_cls.grad, rpn_cls.is_leaf, rpn_cls.grad_fn)
                # print(rpn_reg.requires_grad, rpn_reg.grad, rpn_reg.is_leaf, rpn_reg.grad_fn)
                # print('---grad')
                rpn_scores_raw = rpn_cls[:, :, 0].data
                rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                # proposal layer
                rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                output['rois'] = rois
                output['roi_scores_raw'] = roi_scores_raw
                output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                # if self.training:
                rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
           
            if cfg.RCNN.ENABLED:
                # self.rcnn_net.eval()
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                    rpn_scores_raw = rpn_cls[:, :, 0]
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                    # proposal layer

                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
                    
                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output
