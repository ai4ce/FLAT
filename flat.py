import _init_path
from pgd import PGD
from utils import save_kitti_format, transform_matrix, create_dataloader, create_logger
from evaluate import evaluate
from functools import reduce
from pyquaternion import Quaternion
import tqdm
import re
from datetime import datetime
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils
import argparse
from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
# from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate
from lib.utils.bbox_transform import decode_bbox_target
import tools.train_utils.train_utils as train_utils
from lib.net.point_rcnn_attack import AttackPointRCNN_RPN, AttackPointRCNN_RCNN
from lib.net.point_rcnn import PointRCNN
import torch.nn.functional as F
import torch
import numpy as np
import math
import os


np.random.seed(1024)  # set the same seed


def parse_args():
    parser = argparse.ArgumentParser()
    # FLAT args
    parser.add_argument('--split', default='val_1000',
                        help='The data split for evaluation')
    parser.add_argument('--stage', default='1',
                        help='Attack stage of Point RCNN. Options: "1" for RPN stage, "2" for RCNN stage')
    parser.add_argument('--nb_iter', default=20, type=int,
                        help='Number of attack iterations in PGD')
    parser.add_argument('--task', default='cls',
                        help='Task of attacking. Options: "cls" for classification, "reg" for regression')
    parser.add_argument('--attack_type', default='all',
                        help='Specify attack type. Options: "all", "translation", "rotation"')
    parser.add_argument('--iter_eps', default=0.1, type=float,
                        help='Primary PGD attack step size for each iteration, in translation only/rotation only attacks, this parameter is used.')
    parser.add_argument('--iter_eps2', default=0.01, type=float,
                        help='Secondary PGD attack step size for each iteration, only effective when attack_type is "all" and poly mode is disabled.')
    """
    In our code, iter_eps2 will not effect in translation only/rotation only attack.
    For translation only attack, we specified iter_eps to 0.1
    For rotation only attack, we specified iter_eps to 0.01
    For attacking full trajectory(translation+rotation), we specified iter_eps to 0.01 and iter_eps2 to 0.1
    """
    parser.add_argument('--poly', action='store_true', default=False,
                        help='Polynomial trajectory perturbation option. Notice: if true, attack_type will be fixed(translation)')
    
    

    # PointRCNN args
    parser.add_argument('--cfg_file', type=str, default='./PointRCNN/tools/cfgs/default.yaml',
                        help='specify the config for evaluation')
    parser.add_argument("--eval_mode", type=str, default='rcnn',
                        help="specify the evaluation mode")
    parser.add_argument('--eval_all', action='store_true',
                        default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--test', action='store_true',
                        default=False, help='evaluate without ground truth')
    parser.add_argument("--ckpt", type=str, default='checkpoint_epoch_70.pth',
                        help="specify a checkpoint to be evaluated")
    parser.add_argument("--rpn_ckpt", type=str, default=None,
                        help="specify the checkpoint of rpn if trained separated")
    parser.add_argument("--rcnn_ckpt", type=str, default=None,
                        help="specify the checkpoint of rcnn if trained separated")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument("--extra_tag", type=str, default='nuscenes',
                        help="extra tag for multiple evaluation")
    parser.add_argument('--output_dir', type=str, default=None,
                        help='specify an output directory if needed')
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="specify a ckpt directory to be evaluated if needed")
    parser.add_argument('--save_result', action='store_true',
                        default=False, help='save evaluation results to files')
    parser.add_argument('--save_rpn_feature', action='store_true', default=False,
                        help='save features for separately rcnn training and evaluation')
    parser.add_argument('--random_select', action='store_true',
                        default=True, help='sample to the same number of points')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='ignore the checkpoint smaller than this epoch')
    parser.add_argument("--rcnn_eval_roi_dir", type=str, default=None,
                        help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument("--rcnn_eval_feature_dir", type=str, default=None,
                        help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()
    return args


args = parse_args()


def load_ckpt_based_on_args(model, logger):
    if args.ckpt is not None:
        train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)

    total_keys = model.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
        load_part_ckpt(model, filename=args.rpn_ckpt,
                       logger=logger, total_keys=total_keys)

    if cfg.RCNN.ENABLED and args.rcnn_ckpt is not None:
        load_part_ckpt(model, filename=args.rcnn_ckpt,
                       logger=logger, total_keys=total_keys)


def load_part_ckpt(model, filename, logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info(
            "==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = {
            key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


def eval_one_epoch_joint(model, model_attack, dataloader, epoch_id, result_dir, logger):
    np.random.seed(666)
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    # print(MEAN_SIZE)
    mode = 'TEST' if args.test else 'EVAL'
    poly = args.poly
    task = args.task
    stage = args.stage
    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)

    if args.save_result:
        roi_output_dir = os.path.join(result_dir, 'roi_result', 'data')
        refine_output_dir = os.path.join(result_dir, 'refine_result', 'data')
        rpn_output_dir = os.path.join(result_dir, 'rpn_result', 'data')
        os.makedirs(rpn_output_dir, exist_ok=True)
        os.makedirs(roi_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)

    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch_id)
    logger.info('==> Output file: %s' % result_dir)
    for k, v in model.named_parameters():
        v.requires_grad = False  # fix parameters
    model.eval()
    for k, v in model_attack.named_parameters():
        v.requires_grad = False  # fix parameters
    model_attack.eval()
    for k, v in model_attack.named_parameters():
        if v.requires_grad:
            logger.info('PARAM %s NOT FIXED!', k)

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    dataset = dataloader.dataset
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0

    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval')

    pgd_attack = PGD(model_attack, iter_eps=args.iter_eps, iter_eps2=args.iter_eps2,
                     nb_iter=args.nb_iter, poly=args.poly)
    num_step = 100
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)

    for data in dataloader:
        cnt += 1
        sample_id, pts_rect, pts_features, pts_input, pts_pose = data['sample_id'], data[
            'pts_rect'], data['pts_features'], data['pts_input'], data['pose_lidar']
        batch_size = len(sample_id)
        pose_matrix = np.squeeze(pts_pose)
        # print(pose_matrix)
        #plt.scatter(pts_input[0, :, 0], pts_input[0, :, 2], s=0.2, c='g', alpha=1)

        if not pose_matrix.shape[0] == 2:
            # we firstly transform the pc from kitti coordinate to nuscene coordinate
            start_pc = np.squeeze(pts_input).T
            # <--be careful here! we need to convert to nuscene format
            start_pc = np.dot(
                kitti_to_nu_lidar.rotation_matrix, start_pc[:3, :])

            # change to polar coordinate
            polar_points = np.arctan2(
                start_pc[1, :], start_pc[0, :]) * 180 / np.pi + 180  # in degrees (0, 360]

            polar_points_min = np.floor(np.min(polar_points)-0.1)
            polar_points_max = np.ceil(np.max(polar_points))

            start_pose_rec_translation = [
                pose_matrix[0, 0], pose_matrix[0, 1], pose_matrix[0, 2]]
            start_pose_rec_rotation = [
                pose_matrix[0, 3], pose_matrix[0, 4], pose_matrix[0, 5], pose_matrix[0, 6]]

            start_cs_rec_translation = [
                pose_matrix[1, 0], pose_matrix[1, 1], pose_matrix[1, 2]]
            start_cs_rec_rotation = [
                pose_matrix[1, 3], pose_matrix[1, 4], pose_matrix[1, 5], pose_matrix[1, 6]]

            end_pose_rec_translation = [
                pose_matrix[2, 0], pose_matrix[2, 1], pose_matrix[2, 2]]
            end_pose_rec_rotation = [
                pose_matrix[2, 3], pose_matrix[2, 4], pose_matrix[2, 5], pose_matrix[2, 6]]

            # enable motion distortion
            # Init
            sensor_from_vehicle = transform_matrix(
                start_cs_rec_translation, Quaternion(start_cs_rec_rotation), inverse=True)
            vehicle_from_global = transform_matrix(
                start_pose_rec_translation, Quaternion(start_pose_rec_rotation), inverse=True)

            global_from_car = transform_matrix(
                start_pose_rec_translation, Quaternion(start_pose_rec_rotation), inverse=False)
            car_from_current = transform_matrix(
                start_cs_rec_translation, Quaternion(start_cs_rec_rotation), inverse=False)

            # find the next sample data
            translation_step = (np.array(
                end_pose_rec_translation) - np.array(start_pose_rec_translation))/num_step

            p_start = start_pose_rec_rotation
            q_end = end_pose_rec_rotation

            # trans_matrix_gps_list = list()
            pc_timestap_list = list()

            for t in range(num_step):
                t_current = start_pose_rec_translation + t * translation_step
                q_current = []

                cosa = p_start[0]*q_end[0] + p_start[1]*q_end[1] + \
                    p_start[2]*q_end[2] + p_start[3]*q_end[3]

                # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
                # the shorter path. Fix by reversing one quaternion.
                if cosa < 0.0:
                    q_end[0] = -q_end[0]
                    q_end[1] = -q_end[1]
                    q_end[2] = -q_end[2]
                    q_end[3] = -q_end[3]
                    cosa = -cosa

                # If the inputs are too close for comfort, linearly interpolate
                if cosa > 0.9995:
                    k0 = 1.0 - t/num_step
                    k1 = t/num_step
                else:
                    sina = np.sqrt(1.0 - cosa*cosa)
                    a = math.atan2(sina, cosa)
                    k0 = math.sin((1.0 - t/num_step)*a) / sina
                    k1 = math.sin(t*a/num_step) / sina

                q_current.append(p_start[0]*k0 + q_end[0]*k1)
                q_current.append(p_start[1]*k0 + q_end[1]*k1)
                q_current.append(p_start[2]*k0 + q_end[2]*k1)
                q_current.append(p_start[3]*k0 + q_end[3]*k1)

                ref_from_car = transform_matrix(
                    start_cs_rec_translation, Quaternion(start_cs_rec_rotation), inverse=True)
                car_from_global = transform_matrix(
                    t_current, Quaternion(q_current), inverse=True)

                # select the points in a small scan area
                small_delta = (polar_points_max-polar_points_min)/num_step

                scan_start = polar_points > small_delta*t + polar_points_min
                scan_end = polar_points <= small_delta*(t+1) + polar_points_min
                scan_area = np.logical_and(scan_start, scan_end)
                current_pc = start_pc[:, scan_area]

                # transform point cloud at start timestep into the interpolatation step t
                trans_matrix = reduce(
                    np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                current_pc = trans_matrix.dot(
                    np.vstack((current_pc, np.ones(current_pc.shape[1]))))
                pc_timestap_list.append(current_pc)

                '''
                Now calculate GPS compensation transformation
                '''
                vehicle_from_sensor = transform_matrix(
                    start_cs_rec_translation, Quaternion(start_cs_rec_rotation), inverse=False)
                global_from_vehicle = transform_matrix(
                    t_current, Quaternion(q_current), inverse=False)
                # can also calculate the inverse matrix of trans_matrix
                trans_matrix_gps = reduce(np.dot, [
                                          sensor_from_vehicle, vehicle_from_global, global_from_vehicle, vehicle_from_sensor])

                trans_matrix_gps = np.expand_dims(trans_matrix_gps, 0)

                if t == 0:
                    trans_matrix_gps_tensor = trans_matrix_gps
                else:
                    trans_matrix_gps_tensor = np.concatenate(
                        [trans_matrix_gps_tensor, trans_matrix_gps], 0)  # [1000, 4, 4]
            rpn_cls_label, rpn_reg_label, gt_boxes3d = data[
                'rpn_cls_label'], data['rpn_reg_label'], data['gt_boxes3d']

            rpn_cls_label[rpn_cls_label > -1] = 1 - \
                rpn_cls_label[rpn_cls_label > -1]

            adv_pc = pgd_attack.attack(args.attack_type, pc_timestap_list, trans_matrix_gps_tensor, poly,
                                       stage, task, rpn_cls_label=rpn_cls_label, rpn_reg_label=rpn_reg_label, gt_boxes3d=gt_boxes3d)
            inputs = torch.from_numpy(adv_pc).cuda(non_blocking=True).float()
        else:
            inputs = torch.from_numpy(pts_input).cuda(
                non_blocking=True).float()

        # model inference
        input_data = {'pts_input': inputs}

        # model inference
        ret_dict = model(input_data)

        roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
        roi_boxes3d = ret_dict['rois']  # (B, M, 7)
        seg_result = ret_dict['seg_result'].long()  # (B, N)

        rcnn_cls = ret_dict['rcnn_cls'].view(
            batch_size, -1, ret_dict['rcnn_cls'].shape[1])
        rcnn_reg = ret_dict['rcnn_reg'].view(
            batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

        # bounding box regression
        anchor_size = MEAN_SIZE
        if cfg.RCNN.SIZE_RES_ON_ROI:
            assert False

        pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size=anchor_size,
                                          loc_scope=cfg.RCNN.LOC_SCOPE,
                                          loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                          num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                          get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                          loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                          get_ry_fine=True).view(batch_size, -1, 7)

        # scoring
        if rcnn_cls.shape[2] == 1:
            raw_scores = rcnn_cls  # (B, M, 1)

            norm_scores = torch.sigmoid(raw_scores)
            pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
        else:
            pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
            cls_norm_scores = F.softmax(rcnn_cls, dim=1)
            raw_scores = rcnn_cls[:, pred_classes]
            norm_scores = cls_norm_scores[:, pred_classes]

        # evaluation
        recalled_num = gt_num = rpn_iou = 0
        if not args.test:
            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(
                    rpn_cls_label).cuda(non_blocking=True).long()

            gt_boxes3d = data['gt_boxes3d']

            for k in range(batch_size):
                # calculate recall
                cur_gt_boxes3d = gt_boxes3d[k]
                tmp_idx = cur_gt_boxes3d.__len__() - 1

                while tmp_idx >= 0 and cur_gt_boxes3d[tmp_idx].sum() == 0:
                    tmp_idx -= 1

                if tmp_idx >= 0:
                    cur_gt_boxes3d = cur_gt_boxes3d[:tmp_idx + 1]

                    cur_gt_boxes3d = torch.from_numpy(
                        cur_gt_boxes3d).cuda(non_blocking=True).float()
                    iou3d = iou3d_utils.boxes_iou3d_gpu(
                        pred_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou, _ = iou3d.max(dim=0)
                    refined_iou, _ = iou3d.max(dim=1)

                    for idx, thresh in enumerate(thresh_list):
                        total_recalled_bbox_list[idx] += (
                            gt_max_iou > thresh).sum().item()
                    recalled_num += (gt_max_iou > 0.7).sum().item()
                    gt_num += cur_gt_boxes3d.shape[0]
                    total_gt_bbox += cur_gt_boxes3d.shape[0]

                    # original recall
                    iou3d_in = iou3d_utils.boxes_iou3d_gpu(
                        roi_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou_in, _ = iou3d_in.max(dim=0)

                    for idx, thresh in enumerate(thresh_list):
                        total_roi_recalled_bbox_list[idx] += (
                            gt_max_iou_in > thresh).sum().item()

                if not cfg.RPN.FIXED:
                    fg_mask = rpn_cls_label > 0
                    correct = ((seg_result == rpn_cls_label)
                               & fg_mask).sum().float()
                    union = fg_mask.sum().float() + (seg_result > 0).sum().float() - correct
                    rpn_iou = correct / torch.clamp(union, min=1.0)
                    total_rpn_iou += rpn_iou.item()

        disp_dict = {
            'mode': mode, 'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)}
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

        if args.save_result:
            # save roi and refine results
            roi_boxes3d_np = roi_boxes3d.cpu().numpy()
            pred_boxes3d_np = pred_boxes3d.cpu().numpy()
            roi_scores_raw_np = roi_scores_raw.cpu().numpy()
            raw_scores_np = raw_scores.cpu().numpy()

            rpn_cls_np = ret_dict['rpn_cls'].cpu().numpy()
            rpn_xyz_np = ret_dict['backbone_xyz'].cpu().numpy()
            seg_result_np = seg_result.cpu().numpy()
            output_data = np.concatenate((rpn_xyz_np, rpn_cls_np.reshape(batch_size, -1, 1),
                                          seg_result_np.reshape(batch_size, -1, 1)), axis=2)

            for k in range(batch_size):
                cur_sample_id = sample_id[k]
                calib = dataset.get_calib(cur_sample_id)
                image_shape = dataset.get_image_shape(cur_sample_id)
                save_kitti_format(cur_sample_id, calib, roi_boxes3d_np[k], roi_output_dir,
                                  roi_scores_raw_np[k], image_shape, cfg)
                save_kitti_format(cur_sample_id, calib, pred_boxes3d_np[k], refine_output_dir,
                                  raw_scores_np[k], image_shape, cfg)

                output_file = os.path.join(
                    rpn_output_dir, '%06d.npy' % cur_sample_id)
                np.save(output_file, output_data.astype(np.float32))

        # scores thresh
        inds = norm_scores > cfg.RCNN.SCORE_THRESH

        for k in range(batch_size):
            cur_inds = inds[k].view(-1)
            if cur_inds.sum() == 0:
                continue

            pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
            raw_scores_selected = raw_scores[k, cur_inds]
            norm_scores_selected = norm_scores[k, cur_inds]

            # NMS thresh
            # rotated nms
            boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(
                pred_boxes3d_selected)
            keep_idx = iou3d_utils.nms_gpu(
                boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
            pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
            scores_selected = raw_scores_selected[keep_idx]
            pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu(
            ).detach().numpy(), scores_selected.cpu().detach().numpy()

            cur_sample_id = sample_id[k]
            calib = dataset.get_calib(cur_sample_id)
            final_total += pred_boxes3d_selected.shape[0]
            image_shape = dataset.get_image_shape(cur_sample_id)
            save_kitti_format(cur_sample_id, calib, pred_boxes3d_selected,
                              final_output_dir, scores_selected, image_shape, cfg)

    progress_bar.close()
    # dump empty files
    split_file = os.path.join(dataset.imageset_dir,
                              '..', dataset.split, 'ImageSets', dataset.split + '.txt')
#     print('split_file---', split_file)
    split_file = os.path.abspath(split_file)
#     print('split_file---', split_file)
    image_idx_list = [x.strip() for x in open(split_file).readlines()]
#     print('image_idx_list---', image_idx_list)
    empty_cnt = 0
    for k in range(image_idx_list.__len__()):
        cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
        if not os.path.exists(cur_file):
            with open(cur_file, 'w') as temp_f:
                pass
            empty_cnt += 1
            logger.info('empty_cnt=%d: dump empty file %s' %
                        (empty_cnt, cur_file))

    ret_dict = {'empty_cnt': empty_cnt}

    logger.info(
        '-------------------performance of epoch %s---------------------' % epoch_id)
    logger.info(str(datetime.now()))

    avg_rpn_iou = (total_rpn_iou / max(cnt, 1.0))
    avg_cls_acc = (total_cls_acc / max(cnt, 1.0))
    avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1.0))
    avg_det_num = (final_total / max(len(dataset), 1.0))
    logger.info('final average detections: %.3f' % avg_det_num)
    logger.info('final average rpn_iou refined: %.3f' % avg_rpn_iou)
    logger.info('final average cls acc: %.3f' % avg_cls_acc)
    logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)
    ret_dict['rpn_iou'] = avg_rpn_iou
    ret_dict['rcnn_cls_acc'] = avg_cls_acc
    ret_dict['rcnn_cls_acc_refined'] = avg_cls_acc_refined
    ret_dict['rcnn_avg_num'] = avg_det_num

    for idx, thresh in enumerate(thresh_list):
        cur_roi_recall = total_roi_recalled_bbox_list[idx] / max(
            total_gt_bbox, 1.0)
        logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_roi_recalled_bbox_list[idx],
                                                                          total_gt_bbox, cur_roi_recall))
        ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_roi_recall

    for idx, thresh in enumerate(thresh_list):
        cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_recalled_bbox_list[idx],
                                                                      total_gt_bbox, cur_recall))
        ret_dict['rcnn_recall(thresh=%.2f)' % thresh] = cur_recall

    if cfg.TEST.SPLIT != 'test':
        logger.info('Average Precision:')
        name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        # print(dataset.label_dir, final_output_dir,
        #       split_file, name_to_class[cfg.CLASSES])
        # old eval
        # ap_result_str, ap_dict = kitti_evaluate(dataset.label_dir, final_output_dir, label_split_file=split_file,
        #                                         current_class=name_to_class[cfg.CLASSES])
        # logger.info('old eval:'+ap_result_str)
        # new eval
        ap_result_str, ap_dict = evaluate(dataset.label_dir, final_output_dir, label_split_file=split_file,
                                          current_class=name_to_class[cfg.CLASSES])
        logger.info('new eval:'+ap_result_str)
        ret_dict.update(ap_dict)

    logger.info('result is saved to: %s' % result_dir)
    return ret_dict


def eval_one_epoch(model, model_attack, dataloader, epoch_id, result_dir, logger):
    assert cfg.RPN.ENABLED and cfg.RCNN.ENABLED, 'RPN and RCNN module should be both enabled'
    ret_dict = eval_one_epoch_joint(
        model, model_attack, dataloader, epoch_id, result_dir, logger)
    return ret_dict


if __name__ == '__main__':
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # poly mode only available for translation-only attack
    if args.poly:
        args.attack_type = 'translation'

    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    outputdir = "FLAT_stage{stage}_{task}_{poly}{nb_iter}_{iter_eps}_{iter_eps2}".format(stage=str(args.stage), task=str(args.task), poly=(
        'poly_' if args.poly else ''), nb_iter=str(args.nb_iter), iter_eps=str(args.iter_eps), iter_eps2=str(args.iter_eps2))

    if args.eval_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join(
            'output', args.split, args.attack_type, outputdir)
        ckpt_dir = os.path.join('../', 'output', args.split,
                                args.attack_type, outputdir,  'ckpt')
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir

    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'

    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    # for key, val in vars(args).items():
    #     logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger=logger)

    # create dataloader & network
    test_loader = create_dataloader(logger, args, cfg)
    model = PointRCNN(num_classes=test_loader.dataset.num_class,
                      use_xyz=True, mode='TEST')
    model.cuda()

    if args.stage == '1':
        model_attack = AttackPointRCNN_RPN(
            num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')
    elif args.stage == '2':
        model_attack = AttackPointRCNN_RCNN(
            num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')
    model_attack.cuda()

    # load checkpoint
    load_ckpt_based_on_args(model, logger)

    # start evaluation
    eval_one_epoch(model, model_attack, test_loader,
                   epoch_id, root_result_dir, logger)
