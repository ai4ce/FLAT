import numpy as np
import torch
import torch.nn as nn
from lib.config import cfg
from pyquaternion import Quaternion
from torch.autograd import Variable
import torch.nn.functional as F
import lib.utils.loss_utils as loss_utils

# stage 1


class PGD(nn.Module):
    def __init__(self, model, iter_eps=0.1, iter_eps2=0.01, nb_iter=20, ord=np.inf, rand_init=True, flag_target=False, poly=False):
        super().__init__()
        self.model = model
        self.device = torch.device("cuda" if (
            torch.cuda.is_available()) else "cpu")
        self.iter_eps = iter_eps
        self.iter_eps2 = iter_eps2
        self.nb_iter = nb_iter
        self.clip_value_min = -iter_eps
        self.clip_value_max = iter_eps

        self.clip_value2_min = -iter_eps2
        self.clip_value2_max = iter_eps2


        self.ord = ord
        self.rand_init = rand_init
        self.model.to(self.device)
        self.flag_target = flag_target
        self.num_step = 100
        self.kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        self.kitti_to_nu_lidar_inv = self.kitti_to_nu_lidar.inverse

        self.poly = poly

    def get_pc_ori(self, pc_timestap_list, trans_matrix_gps_tensor):

        trans_matrix_gps_tensor = torch.Tensor(
            trans_matrix_gps_tensor).to(self.device)
        # aggregate the motion distortion points
        init_flag = False
        for timestap in range(self.num_step):
            pc_curr = pc_timestap_list[timestap]

            if not pc_curr.shape[1] == 0:
                pc_curr = torch.Tensor(pc_curr).to(self.device)
                tmp_pc = torch.mm(
                    trans_matrix_gps_tensor[timestap, :], pc_curr)[:3, :]
                if init_flag == False:
                    all_pc = tmp_pc
                    init_flag = True
                else:
                    all_pc = torch.cat((all_pc, tmp_pc), 1)  # 3 * 16384

        KITTI_to_NU_R = Variable(torch.Tensor(
            self.kitti_to_nu_lidar_inv.rotation_matrix))
        KITTI_to_NU_R = KITTI_to_NU_R.to(self.device)

        inputs = torch.mm(KITTI_to_NU_R, all_pc)
        ori_pc = torch.unsqueeze(inputs.transpose(0, 1), 0)

        return ori_pc


    def get_pc(self, attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly, **params):

        trans_matrix_gps_tensor = torch.Tensor(trans_matrix_gps_tensor).to(self.device)

        
        if poly:
            x = params['x']
            poly_x = params['poly_x']
            poly_beta = params['poly_beta']
            clip = False
            poly_beta_tensor = Variable(torch.Tensor(
                poly_beta).to(self.device), requires_grad=True)
            poly_x_tensor = torch.Tensor(poly_x.T).to(self.device)
            x = torch.Tensor(x).to(self.device)
            adv_poly_pertubation = torch.mm(poly_x_tensor, poly_beta_tensor)

            if(clip):
                # print(adv_poly_pertubation)
                adv_poly_pertubation = torch.clamp(
                    adv_poly_pertubation, self.clip_min, self.clip_max)
                # print(adv_poly_pertubation)
            adv_pertubation = adv_poly_pertubation + x

        else:
            adv_pertubation = params['adv_pertubation']
            adv_pertubation = Variable(torch.Tensor(
                adv_pertubation),  requires_grad=True)

        if attack_type == 'translation':
            trans_matrix_gps_tensor[:, :3, 3] = adv_pertubation
        elif attack_type == 'rotation':
            trans_matrix_gps_tensor[:, :3, :3] = adv_pertubation
        elif attack_type == 'all':
            trans_matrix_gps_tensor[:, :3, :] = adv_pertubation


        # aggregate the motion distortion points
        init_flag = False
        for timestap in range(self.num_step):
            pc_curr = pc_timestap_list[timestap]

            if not pc_curr.shape[1] == 0:
                pc_curr = torch.Tensor(pc_curr).to(self.device)
                tmp_pc = torch.mm(
                    trans_matrix_gps_tensor[timestap, :], pc_curr)[:3, :]
                if init_flag == False:
                    all_pc = tmp_pc
                    init_flag = True
                else:
                    all_pc = torch.cat((all_pc, tmp_pc), 1)  # 3 * 16384

        KITTI_to_NU_R = Variable(torch.Tensor(
            self.kitti_to_nu_lidar_inv.rotation_matrix))
        KITTI_to_NU_R = KITTI_to_NU_R.to(self.device)

        inputs = torch.mm(KITTI_to_NU_R, all_pc)
        adv_pc = torch.unsqueeze(inputs.transpose(0, 1), 0)

        if poly:
            return adv_pc, poly_beta_tensor, adv_poly_pertubation, adv_pertubation
        else:
            return adv_pc, adv_pertubation   

    def single_step_attack(self, attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly, stage, task, x, **params):
        


        if poly:
            poly_x = params['poly_x']
            poly_beta = params['poly_beta']
            ori_pc = params['ori_pc']
            adv_pc, adv_x, _, _ = self.get_pc(
                attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly=True, x=x, poly_x=poly_x, poly_beta=poly_beta, clip=False)
        # get adversarial  point cloud
        else:
            pertubation = params['pertubation']
            adv_pc, adv_x = self.get_pc(
                attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly=False, adv_pertubation=x + pertubation)

        assert adv_pc.shape[1] == 16384, 'point cloud has changed!'

        tb_dict = {}

        if stage == '2':
            # data = params['data']
            gt_boxes3d = torch.from_numpy(params['gt_boxes3d']).cuda(non_blocking=True).float()
            input_data = {'pts_input': adv_pc, 'gt_boxes3d': gt_boxes3d}
            ret_dict = self.model(input_data)

            if task == 'cls':
                loss = self.get_rcnn_cls_loss(self.model, ret_dict, tb_dict)
            elif task == 'reg':
                loss = self.get_rcnn_reg_loss(self.model, ret_dict, tb_dict)

        elif stage == '1':
            rpn_cls_label, rpn_reg_label = params['rpn_cls_label'], params['rpn_reg_label']

            input_data = {'pts_input': adv_pc}
            ret_dict = self.model(input_data)
            # calculate the adversarial loss

            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']

            if task == 'cls':
                loss = self.get_rpn_cls_loss(self.model, rpn_cls, rpn_cls_label, tb_dict)
            elif task == 'reg':
                loss = self.get_rpn_reg_loss(rpn_reg, rpn_cls_label, rpn_reg_label)
        
        # backpropogate the loss to the pertubation
        if poly:
            lp_loss = torch.max(torch.abs(torch.sqrt(
                torch.sum((adv_pc - ori_pc) * (adv_pc - ori_pc), 2) + 1e-16)))

            if(lp_loss > 0.1):
                loss = loss - 5e-4 * (lp_loss - 0.1)

        self.model.zero_grad()
        loss.backward()


        grad = adv_x.grad.data
        grad_cpu = grad.cpu().detach().numpy()

        if poly:
            adv_x = adv_x.cpu().detach().numpy() + self.iter_eps * np.sign(grad_cpu)

            pertubation = adv_x
        else:
            if attack_type == 'all':
                pertubation[:, :, :3] = self.iter_eps * \
                    np.sign(grad_cpu[:, :, :3])
                pertubation[:, :, 3] = self.iter_eps2 * \
                    np.sign(grad_cpu[:, :, 3])
            else:
                pertubation = self.iter_eps * np.sign(grad_cpu)

            adv_x = adv_x.cpu().detach().numpy() + pertubation

            if attack_type == 'all':
                pertubation[:, :, :3] = np.clip(
                    adv_x[:, :, :3] - x[:, :, :3], self.clip_value_min, self.clip_value_max)
                pertubation[:, :, 3] = np.clip(
                    adv_x[:, :, 3] - x[:, :, 3], self.clip_value2_min, self.clip_value2_max)
            else:
                pertubation = np.clip(
                    adv_x - x, self.clip_value_min, self.clip_value_max)

        return pertubation

    def attack(self, attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly, stage, task, **params):
        # data = params['data']
        rpn_cls_label, rpn_reg_label = params['rpn_cls_label'], params['rpn_reg_label']
        # TODO: Check this usage for?
        # rpn_cls_label[rpn_cls_label > -1] = 1 - \
        #     rpn_cls_label[rpn_cls_label > -1]

        rpn_cls_label = torch.Tensor(rpn_cls_label).to(self.device)
        rpn_reg_label = torch.Tensor(rpn_reg_label).to(self.device)

        trans_matrix_gps_tensor = trans_matrix_gps_tensor

        if attack_type == 'translation':
            x = trans_matrix_gps_tensor[:, :3, 3]
            if poly:
                base_x = np.linspace(-1, 1, 100)
            else:
                pertubation = np.zeros(x.shape)
        elif attack_type == 'rotation':
            x = trans_matrix_gps_tensor[:, :3, :3]
            pertubation = np.zeros(x.shape)
        elif attack_type == 'all':
            x = trans_matrix_gps_tensor[:, :3, :]
            pertubation = np.zeros(x.shape)

        if poly:
            poly_degree = 3
            poly_beta = 1e-8 * np.ones((poly_degree+1, 3))
            base_list = []
            for i in range(poly_degree+1):
                base_list.append(np.power(base_x, i))
            poly_x = np.array(base_list, dtype=np.float)
            ori_pc = self.get_pc_ori(
                pc_timestap_list, trans_matrix_gps_tensor)
                
        gt_boxes3d = params['gt_boxes3d']
        for i in range(self.nb_iter):
            if poly:
                # ori_pc, x, poly_x, poly_beta,
                poly_beta = self.single_step_attack(
                    attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly=True, stage = stage, task = task, x=x, ori_pc=ori_pc, poly_x=poly_x, poly_beta=poly_beta, rpn_cls_label = rpn_cls_label, rpn_reg_label = rpn_reg_label, gt_boxes3d = gt_boxes3d)
            else:
                # x, pertubation,
                pertubation = self.single_step_attack(
                    attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly=False, stage = stage, task = task, x=x, pertubation=pertubation, rpn_cls_label = rpn_cls_label, rpn_reg_label = rpn_reg_label, gt_boxes3d = gt_boxes3d)

        if poly:
            adv_pc, _, adv_poly_pertubation, adv_pertubation = self.get_pc(
                attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly=True, x=x, poly_x=poly_x, poly_beta=poly_beta, clip=False)
        
        else:
            if attack_type == 'all':
                pertubation[:, :, :3] = np.clip(
                    pertubation[:, :, :3], self.clip_value_min, self.clip_value_max)
                pertubation[:, :, 3] = np.clip(
                    pertubation[:, :, 3], self.clip_value2_min, self.clip_value2_max)
            else:
                pertubation = np.clip(
                    pertubation, self.clip_value_min, self.clip_value_max)
            adv_x = x + pertubation
            adv_pc, _ = self.get_pc(
                attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly = False, adv_pertubation = adv_x)
            
        return adv_pc.cpu().detach().numpy()


    # loss functions

    def get_rpn_reg_loss(self, rpn_reg, rpn_cls_label, rpn_reg_label):


        rpn_cls_label_flat = rpn_cls_label.view(-1)

        fg_mask = (rpn_cls_label_flat > 0)

        MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

        # # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 7)[
                    fg_mask],
                    loc_scope=cfg.RPN.LOC_SCOPE,
                    loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                    num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                    anchor_size=MEAN_SIZE,
                    get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                    get_y_by_bin=False,
                    get_ry_fine=False)

            loss_size = 3 * loss_size  # consistent with old codes
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            # TODO: This code is buggy, check later
            loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = - rpn_loss_reg

        return rpn_loss


    def get_rpn_cls_loss(self, model, rpn_cls, rpn_cls_label, tb_dict):
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)

        rpn_cls_flat = rpn_cls.view(-1)

        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
            #print(rpn_cls_flat.shape, rpn_cls_target.shape)
            rpn_loss_cls = rpn_cls_loss_func(
                rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight=weight, reduction='none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / \
                torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        return rpn_loss_cls

    def get_rcnn_cls_loss(self, model, ret_dict, tb_dict):

        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']
        cls_label = ret_dict['cls_label'].float()
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']

        # print(cls_label)
        # cls_label[cls_label>-1] = 1 - cls_label[cls_label>-1]
        # print(cls_label)

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func

        cls_label_flat = cls_label.view(-1)
        #print(cls_label_flat)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)

            rcnn_loss_cls = cls_loss_func(
                rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            # print(rcnn_cls_flat)
            batch_loss_cls = F.binary_cross_entropy(
                torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / \
                torch.clamp(cls_valid_mask.sum(), min=1.0)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()
            cls_valid_mask = (cls_label_flat >= 0).float()

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = torch.clamp(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim=1) *
                             cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        return rcnn_loss_cls

    def get_rcnn_reg_loss(self, model, ret_dict, tb_dict):

        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']
        cls_label = ret_dict['cls_label'].float()
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func

        MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

        cls_label_flat = cls_label.view(-1)
        #print(cls_label_flat)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)
            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)

            rcnn_loss_cls = cls_loss_func(
                rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            # print(rcnn_cls_flat)
            batch_loss_cls = F.binary_cross_entropy(
                torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / \
                torch.clamp(cls_valid_mask.sum(), min=1.0)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()
            cls_valid_mask = (cls_label_flat >= 0).float()

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = torch.clamp(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim=1) *
                             cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 7)[
                    fg_mask],
                    loc_scope=cfg.RCNN.LOC_SCOPE,
                    loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                    num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                    anchor_size=anchor_size,
                    get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                    loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                    get_ry_fine=True)

            loss_size = 3 * loss_size  # consistent with old codes
            rcnn_loss_reg = loss_loc + loss_angle + loss_size
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = rcnn_loss_reg = rcnn_loss_cls * 0

        rcnn_loss = -rcnn_loss_reg
        return rcnn_loss
