# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Modified by Congcong Wen, Yiming Li and Qi Fang
"""
This script converts nuScenes data to KITTI format and KITTI results to nuScenes.
It is used for compatibility with software that uses KITTI-style annotations.

We modified the original nuScenes script for creating our own nuscenes dataset in kitti format, along with pose. 

This script includes function:
- nuscenes_gt_to_kitti(): Converts nuScenes GT annotations to KITTI format.

To launch these scripts run:
- python nusc_to_kitti.py nuscenes_gt_to_kitti


See https://www.nuscenes.org/object-detection for more information on the nuScenes result format.
"""

import os
from typing import List

import fire
import numpy as np
import random
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix, view_points
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs

KITTI_CATEGORY = ['car', 'van', 'truck', 'pedestrian',
                  'person_sitting', 'cyclist', 'tram', 'misc', 'dontCare']

random.seed(1024)


class KittiConverter:
    def __init__(self,
                 nusc_kitti_dir: str = './dataset/nusc_kitti',
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                 image_count: int = 1000,
                 nusc_version: str = 'v1.0-trainval',
                 dataroot: str = '/data2/yimingli/nuscenes/v1.0-trainval',
                 shuffle : bool = False,
                 split: str = 'val'):
        """
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param dataroot: nuScenes dataset path.
        :param shuffle: Whether to shuffle samples of nuscense.
        :param split: Dataset split to use(train/val/mini_train/mini_val).
        """
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.image_count = image_count
        self.nusc_version = nusc_version
        self.split = split
        self.shuffle = shuffle
        
        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes(
            version=nusc_version, dataroot=dataroot)

    def project_to_2d(self, box, p_left, height, width):
        box = box.copy()

        # KITTI defines the box center as the bottom center of the object.
        # We use the true center, so we need to adjust half height in negative y direction.
        box.translate(np.array([0, -box.wlh[2] / 2, 0]))

        # Check that some corners are inside the image.
        corners = np.array(
            [corner for corner in box.corners().T if corner[2] > 0]).T
        if len(corners) == 0:
            return None, None

        # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
        imcorners = view_points(corners, p_left, normalize=True)[:2]
        bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(
            imcorners[0]), np.max(imcorners[1]))

        inside = (0 <= bbox[1] < height and 0 < bbox[3] <= height) and (
            0 <= bbox[0] < width and 0 < bbox[2] <= width)
        valid = (0 <= bbox[1] < height or 0 < bbox[3] <= height) and (
            0 <= bbox[0] < width or 0 < bbox[2] <= width)
        if not valid:
            return None, None

        truncated = valid and not inside
        if truncated:
            _bbox = [0] * 4
            _bbox[0] = max(0, bbox[0])
            _bbox[1] = max(0, bbox[1])
            _bbox[2] = min(width, bbox[2])
            _bbox[3] = min(height, bbox[3])

            truncated = 1.0 - ((_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])
                               ) / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            bbox = _bbox
        else:
            truncated = 0.0
        return bbox, truncated

    def get_alpha(self, box_cam_kitti):
        x, _, z = box_cam_kitti.center
        yaw, _, _ = box_cam_kitti.orientation.yaw_pitch_roll
        yaw = -yaw
        alpha = yaw - np.arctan2(x, z)
        return alpha

    def write_occlusion(self, objs, height, width):
        _map = np.ones((height, width), dtype=np.int8) * -1
        objs = sorted(objs, key=lambda x: x["depth"], reverse=True)
        for i, obj in enumerate(objs):
            _map[int(round(obj["bbox_2d"][1])):int(round(obj["bbox_2d"][3])), int(
                round(obj["bbox_2d"][0])):int(round(obj["bbox_2d"][2]))] = i
        unique, counts = np.unique(_map, return_counts=True)
        counts = dict(zip(unique, counts))
        for i, obj in enumerate(objs):
            if i not in counts.keys():
                counts[i] = 0
            occlusion = 1.0 - counts[i] / (obj["bbox_2d"][3] - obj["bbox_2d"][1]) / (
                obj["bbox_2d"][2] - obj["bbox_2d"][0])
            obj["occluded"] = int(np.clip(occlusion * 4, 0, 3))
        return objs

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        token_idx = 0  # Start tokens from 0.

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)
        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        # shuffle option
        if self.shuffle:
            random.shuffle(sample_tokens)
        
        tokens = []

        self.split = self.split + "_" + str(self.image_count)

        # Create output folders.
        label_folder = os.path.join(self.nusc_kitti_dir, self.split, 'label_2')
        calib_folder = os.path.join(self.nusc_kitti_dir, self.split, 'calib')
        image_folder = os.path.join(self.nusc_kitti_dir, self.split, 'image_2')
        lidar_folder = os.path.join(self.nusc_kitti_dir, self.split, 'velodyne')
        pose_folder = os.path.join(self.nusc_kitti_dir, self.split, 'pose')
        split_folder = os.path.join(self.nusc_kitti_dir, self.split, 'ImageSets')

        for folder in [label_folder, calib_folder, image_folder, lidar_folder, pose_folder, split_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        split_file_path = os.path.join(split_folder, self.split + '.txt')
        split_file = open(split_file_path, "w")

        for sample_token in sample_tokens:
            # Get sample data.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']
            cam_front_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]

            # Retrieve sensor records.
            sd_record_cam = self.nusc.get('sample_data', cam_front_token)
            sd_record_lid = self.nusc.get('sample_data', lidar_token)
            cs_record_cam = self.nusc.get(
                'calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            cs_record_lid = self.nusc.get(
                'calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

            # We use ego pose of current frame and adjacent frame for interpolation. Then simulate distorted point cloud.
            ego_pose = self.nusc.get(
                'ego_pose', sd_record_lid['ego_pose_token'])
            # Ego pose(translation+rotation) in current frame.
            start_pose_rec = self.format_list_float_06(
                ego_pose['translation'] + ego_pose['rotation'])
            # Calib for ego pose - lidar sensor in current frame.
            start_cs_rec_lid = self.format_list_float_06(
                cs_record_lid['translation'] + cs_record_lid['rotation'])
            
            # Try to get pose of adjacent frame
            try:
                next_sd_record_lid = self.nusc.get(
                    'sample_data', sd_record_lid['next'])
                next_cs_record_lid = self.nusc.get(
                    'calibrated_sensor', next_sd_record_lid['calibrated_sensor_token'])
                next_ego_pose = self.nusc.get(
                    'ego_pose', next_sd_record_lid['ego_pose_token'])
                end_pose_rec = self.format_list_float_06(
                    next_ego_pose['translation'] + next_ego_pose['rotation'])
                end_cs_rec_lid = self.format_list_float_06(
                    next_cs_record_lid['translation'] + next_cs_record_lid['rotation'])
            except:
                # Last frame in lidar sequence. set adjacent pose to blank.
                end_pose_rec = ''
                end_cs_rec_lid = ''


            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                          inverse=False)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                          inverse=True)
            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(
                velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            # Cameras are always rectified.
            p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            assert (velo_to_cam_rot.round(0) == np.array(
                [[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
            assert (velo_to_cam_trans[1:3] < 0).all()

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            token = '%06d' % token_idx  # use KITTI names.
            token_idx += 1

            # Convert image (jpg to png). Currently not needed.
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(image_folder, token + '.png')
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            dst_lid_path = os.path.join(lidar_folder, token + '.bin')
            assert not dst_lid_path.endswith('.pcd.bin')
            pcl = LidarPointCloud.from_file(src_lid_path)
            # In KITTI lidar frame.
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to tokens.
            tokens.append(sample_token)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
            kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
            # Cameras are already rectified.
            kitti_transforms['R0_rect'] = r0_rect.rotation_matrix
            kitti_transforms['Tr_velo_to_cam'] = np.hstack(
                (velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
            calib_path = os.path.join(calib_folder, token + '.txt')
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))

            # Write split file
            split_file.write(token+'\n')

            # Write label file.
            label_path = os.path.join(label_folder, token + '.txt')
            if os.path.exists(label_path):
                print('Skipping existing file: %s' % label_path)
                continue
            else:
                print('Writing file: %s' % label_path)
            with open(label_path, "w") as label_file:
                objs = []
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = self.nusc.get(
                        'sample_annotation', sample_annotation_token)

                    # Get box in LIDAR frame.
                    _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                     selected_anntokens=[sample_annotation_token])

                    box_lidar_nusc = box_lidar_nusc[0]

                    obj = {}

                    # Convert nuScenes category to nuScenes detection challenge category.
                    detection_name = category_to_detection_name(
                        sample_annotation['category_name'])

                    # Skip categories that are not part of the nuScenes detection challenge.
                    if detection_name is None or not detection_name in KITTI_CATEGORY:
                        continue

                    # Convert from nuScenes to KITTI box format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Convert alpha angle
                    alpha = self.get_alpha(box_cam_kitti)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d, truncated = self.project_to_2d(
                        box_cam_kitti, p_left_kitti, imsize[1], imsize[0])
                    if bbox_2d is None:
                        continue

                    # Compute depth for occlusion
                    depth = np.linalg.norm(np.array(box_cam_kitti.center[:3]))

                    obj["detection_name"] = detection_name.capitalize()
                    obj["box_cam_kitti"] = box_cam_kitti
                    obj["alpha"] = alpha
                    obj["bbox_2d"], obj["truncated"] = bbox_2d, truncated
                    obj["depth"] = depth
                    objs.append(obj)

                # Compute occlusion
                objs = self.write_occlusion(objs, imsize[1], imsize[0])

                for obj in objs:
                    # Convert box to output string format.
                    output = KittiDB.box_to_string(name=obj["detection_name"], box=obj["box_cam_kitti"], bbox_2d=obj["bbox_2d"],
                                                   truncation=obj["truncated"], occlusion=obj["occluded"], alpha=obj["alpha"])

                    # Write to disk.
                    label_file.write(output + '\n')

            # Pose output
            if end_pose_rec != '':
                pose_path = os.path.join(pose_folder, token + '.txt')
                with open(pose_path, "w") as pose_file:
                    pose_file.write('%s\n' % ','.join(start_pose_rec))
                    pose_file.write('%s\n' % ','.join(start_cs_rec_lid))
                    pose_file.write('%s\n' % ','.join(end_pose_rec))
                    pose_file.write('%s\n' % ','.join(end_cs_rec_lid))
            else:
                # Frame without adjacent pose won't save pose record.
                print("End of lidar pose sequence in %s." %
                      token, "Skip writing pose file.")

    def format_list_float_06(self, l) -> None:
        for index, value in enumerate(l):
            l[index] = '%.6f' % value
        return l

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples


if __name__ == '__main__':
    fire.Fire(KittiConverter)
