import os
import os.path as osp
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
import trimesh
import pickle
import PIL
import tyro
import smplx
import copy
import time
import tqdm
import json

from typing import List
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.optim_factory import adjust_learning_rate_by_lr
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt.commons import cosine_schedule, linear_schedule

from vis_viser_hsfm import show_env_human_in_viser, show_optimization_results
from joint_names import COCO_WHOLEBODY_KEYPOINTS, ORIGINAL_SMPLX_JOINT_NAMES, COCO_MAIN_BODY_SKELETON, SMPL_45_KEYPOINTS

coco_main_body_end_joint_idx = COCO_WHOLEBODY_KEYPOINTS.index('right_heel') 
coco_main_body_joint_idx = list(range(coco_main_body_end_joint_idx + 1))
coco_main_body_joint_names = COCO_WHOLEBODY_KEYPOINTS[:coco_main_body_end_joint_idx + 1]
smplx_main_body_joint_idx = [ORIGINAL_SMPLX_JOINT_NAMES.index(joint_name) for joint_name in coco_main_body_joint_names] 
smpl_main_body_joint_idx = [SMPL_45_KEYPOINTS.index(joint_name) for joint_name in coco_main_body_joint_names]


class Timer:
    def __init__(self):
        self.times = []
        self.start_time = None

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        if self.start_time is None:
            raise RuntimeError("Timer.tic() must be called before Timer.toc()")
        self.times.append(time.time() - self.start_time)
        self.start_time = None

    @property
    def average_time(self):
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)
    
    @property
    def total_time(self):
        return sum(self.times)

def draw_2d_keypoints(img, keypoints, keypoints_name=None, color=(0, 255, 0), radius=1):
    for i, keypoint in enumerate(keypoints):
        img = cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), radius, color, -1)
    return img

def get_dust3r_init_data(results, device='cuda'):
    cam_names = sorted(list(results.keys()))
    pts3d = [torch.from_numpy(results[img_name]['pts3d']).to(device) for img_name in cam_names]
    im_focals = [results[img_name]['intrinsic'][0][0] for img_name in cam_names]
    im_poses = [torch.from_numpy(results[img_name]['cam2world']).to(device) for img_name in cam_names]
    im_poses = torch.stack(im_poses)

    affine_matrix_list = [results[img_name]['affine_matrix'] for img_name in cam_names]
    rgbimg_list = [results[img_name]['rgbimg'] for img_name in cam_names]

    return cam_names, pts3d, im_focals, im_poses, affine_matrix_list, rgbimg_list

def get_smplx_init_data(data_path, frame_list, body_model_name):
    """
    get the smpl(x) params from the data_path, which is the directory path to the smpl_params_xxxxx.pkl
    """

    smplx_params_dict = {}
    for frame_idx in frame_list:
        # smplx data path is like this: smplx_params_00001.pkl
        if body_model_name == 'smplx':
            smplx_data_path = osp.join(data_path, f'smplx_params_{frame_idx:05d}.pkl')
        else:
            smplx_data_path = osp.join(data_path, f'smpl_params_{frame_idx:05d}.pkl')
        if not osp.exists(smplx_data_path):
            continue
        with open(smplx_data_path, 'rb') as f:
            smplx_param = pickle.load(f)
            # For smpl_params, 
            # keys are person_ids and values are 'smpl_params': {'global_orient': (1, 3, 3), 'body_pose': (23, 3, 3), 'betas': (10)}
            # For smplx_params, 
            smplx_params_dict[frame_idx] = smplx_param

    return smplx_params_dict

def get_pose2d_init_data(data_path, frame_list):
    """
    get the pose2d params from the data_path, which is the directory path to the pose2d_xxxxx.pkl
    """
    pose2d_params_dict = {}

    for frame_idx in frame_list:
        pose2d_data_path = osp.join(data_path, f'pose_{frame_idx:05d}.json')
        if not osp.exists(pose2d_data_path):
            continue
        with open(pose2d_data_path, 'r') as f:
            pose2d_params = json.load(f)
            # keys are person_ids and values are 'keypoints', 'bbox'
            # change person id string to int
            pose2d_params_dict[frame_idx] = {int(person_id): pose2d_params[person_id] for person_id in pose2d_params}

    return pose2d_params_dict

def get_bbox_init_data(data_path, frame_list):
    """
    get the bbox params from the data_path, which is the directory path to the mask_xxxxx.json
    """
    bbox_params_dict = {}
    for frame_idx in frame_list:
        bbox_data_path = osp.join(data_path, f'mask_{frame_idx:05d}.json')
        if not osp.exists(bbox_data_path):
            continue
        with open(bbox_data_path, 'r') as f:
            bbox_params = json.load(f)
            # {"mask_name": "mask_00215.npy", "mask_height": 1280, "mask_width": 720, "promote_type": "mask", "labels": {"2": {"instance_id": 2, "class_name": "person", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "logit": 0.0}, "1": {"instance_id": 1, "class_name": "person", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "logit": 0.0}}}
        # keys should be person_ids
        new_bbox_params = {}
        for person_id, person_info in bbox_params['labels'].items():
            bbox_xyxy = np.array([person_info['x1'], person_info['y1'], person_info['x2'], person_info['y2']])
            # sanity check
            if sum(bbox_xyxy) == 0:
                continue

            new_bbox_params[int(person_id)] = bbox_xyxy

        bbox_params_dict[frame_idx] = new_bbox_params

    return bbox_params_dict

def get_mask_init_data(data_path, frame_list):
    """
    get the mask params from the data_path, which is the directory path to the mask_xxxxx.npy
    """
    mask_params_dict = {}
    for frame_idx in frame_list:
        mask_data_path = osp.join(data_path, f'mask_{frame_idx:05d}.npy')
        if not osp.exists(mask_data_path):
            continue
        mask_params = np.load(mask_data_path)
        mask_params_dict[frame_idx] = mask_params # (H, W)

    return mask_params_dict


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def torch_angle_axis_to_rotation_matrix(angle_axis):
    """Convert angle-axis representation to rotation matrix using torch"""
    angle = torch.norm(angle_axis)
    if angle < 1e-7:
        return torch.eye(3, device=angle_axis.device)
    
    axis = angle_axis / angle
    K = torch.zeros((3,3), device=angle_axis.device)
    K[0,1] = -axis[2]
    K[0,2] = axis[1]
    K[1,0] = axis[2]
    K[1,2] = -axis[0]
    K[2,0] = -axis[1]
    K[2,1] = axis[0]
    
    rotation_matrix = torch.eye(3, device=angle_axis.device) + \
                     torch.sin(angle) * K + \
                     (1 - torch.cos(angle)) * torch.matmul(K, K)
    
    return rotation_matrix

def adjust_lr(cur_iter, niter, lr_base, lr_min, optimizer, schedule):
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    else:
        raise ValueError(f'bad lr {schedule=}')
    adjust_learning_rate_by_lr(optimizer, lr)
    return lr
    
def parse_to_save_data(scene, cam_names, main_cam_idx=None):
    # Get optimized values from scene
    pts3d = scene.get_pts3d()
    depths = scene.get_depthmaps()
    msk = scene.get_masks()
    confs = [c for c in scene.im_conf]
    intrinsics = scene.get_intrinsics()
    cams2world = scene.get_im_poses()

    # Convert to numpy arrays
    intrinsics = to_numpy(intrinsics)
    cams2world = to_numpy(cams2world)
    pts3d = to_numpy(pts3d)
    depths = to_numpy(depths)
    msk = to_numpy(msk)
    confs = to_numpy(confs)
    rgbimg = scene.imgs

    if main_cam_idx is not None:
        main_cam_cam2world = cams2world[main_cam_idx]
        # transform all the cameras and pts3d with the transformation matrix, which is the inverse of the main cam extrinsic matrix
        main_cam_world2cam = np.linalg.inv(main_cam_cam2world)
        for i, cam_name in enumerate(cam_names):
            cams2world[i] = main_cam_world2cam @ cams2world[i]
            pts3d[i] =  pts3d[i] @ main_cam_world2cam[:3, :3].T + main_cam_world2cam[:3, 3:].T

    # Save the results as a pickle file
    results = {}
    for i, cam_name in enumerate(cam_names):
        results[cam_name] = {
            'rgbimg': rgbimg[i],
            'intrinsic': intrinsics[i],
            'cam2world': cams2world[i],
            'pts3d': pts3d[i],
            'depths': depths[i],
            'msk': msk[i],
            'conf': confs[i],
        }
    return results

def estimate_initial_trans(joints3d, joints2d, focal, princpt, skeleton, device='cuda'):
    """
    use focal length and bone lengths to approximate distance from camera
    joints3d: (J, 3), xyz in meters
    joints2d: (J, 2+1), xy pixels + confidence
    focal: scalar
    princpt: (2,), x, y
    skeleton: list of edges (bones)

    returns:
        init_trans: (3,), x, y, z in meters, translation vector of the pelvis (root) joint
    """
    # Convert inputs to torch tensors if they aren't already
    # joints3d = torch.as_tensor(joints3d, device=device)
    # joints2d = torch.as_tensor(joints2d, device=device)
    # focal = torch.as_tensor(focal, device=device)
    # princpt = torch.as_tensor(princpt, device=device)

    # Calculate bone lengths and confidence for each bone
    bone3d_array = []  # 3D bone lengths in meters
    bone2d_array = []  # 2D bone lengths in pixels
    conf2d = []        # Confidence scores for each bone

    for edge in skeleton:
        # 3D bone length
        joint1_3d = joints3d[edge[0]]
        joint2_3d = joints3d[edge[1]]
        bone_length_3d = torch.norm(joint1_3d - joint2_3d)
        bone3d_array.append(bone_length_3d)

        # 2D bone length
        joint1_2d = joints2d[edge[0], :2]  # xy coordinates
        joint2_2d = joints2d[edge[1], :2]  # xy coordinates
        bone_length_2d = torch.norm(joint1_2d - joint2_2d)
        bone2d_array.append(bone_length_2d)

        # Confidence score for this bone (minimum of both joint confidences)
        bone_conf = torch.min(joints2d[edge[0], 2], joints2d[edge[1], 2])
        conf2d.append(bone_conf)

    # Convert to torch tensors
    bone3d_array = torch.stack(bone3d_array)
    bone2d_array = torch.stack(bone2d_array)
    conf2d = torch.stack(conf2d)

    mean_bone3d = torch.mean(bone3d_array)
    mean_bone2d = torch.mean(bone2d_array * (conf2d > 0.0))

    # Estimate z using the ratio of 3D to 2D bone lengths
    # z = f * (L3d / L2d) where f is focal length
    z = mean_bone3d / mean_bone2d * focal
    
    # Find pelvis (root) joint position in 2D
    pelvis_2d = joints2d[0, :2]  # Assuming pelvis is the first joint

    # Back-project 2D pelvis position to 3D using estimated z
    x = (pelvis_2d[0] - princpt[0]) * z / focal
    y = (pelvis_2d[1] - princpt[1]) * z / focal

    init_trans = torch.stack([x, y, z])
    return init_trans
    

def init_human_params(smplx_layer, body_model_name, multiview_multiple_human_cam_pred, multiview_multiperson_pose2d, focal_length, princpt, device='cuda', get_vertices=False):
    # multiview_multiple_human_cam_pred: Dict[camera_name -> Dict[human_name -> 'pose2d', 'bbox', 'params' Dicts]]
    # multiview_multiperson_pose2d: Dict[human_name -> Dict[cam_name -> (J, 2+1)]] torch tensor
    # focal_length: scalar, princpt: (2,), device: str

    # Initialize Stage 1: Get the 3D root translation of all humans from all cameras
    # Decode the smplx mesh and get the 3D bone lengths / compare them with the bone lengths from the vitpose 2D bone lengths
    camera_names = sorted(list(multiview_multiple_human_cam_pred.keys()))
    first_cam = camera_names[0]
    first_cam_human_name_counts = {human_name: {'count': 0, 'pose2d_conf': 0} for human_name in sorted(list(multiview_multiple_human_cam_pred[first_cam].keys()))}
    missing_human_names_in_first_cam = defaultdict(list)
    multiview_multiperson_init_trans = defaultdict(dict) # Dict[human_name -> Dict[cam_name -> (3)]]
    for cam_name in camera_names:
        for human_name in sorted(list(multiview_multiple_human_cam_pred[cam_name].keys())):
            params = multiview_multiple_human_cam_pred[cam_name][human_name]['params']
            body_pose = params['body_pose'] #.reshape(1, -1).to(device)
            global_orient = params['global_orient'] #.reshape(1, -1).to(device)
            betas = params['betas'] #.reshape(1, -1).to(device)


            if body_model_name == 'smplx':
                body_pose = body_pose.reshape(1, -1).to(device)
                betas = betas.reshape(1, -1).to(device)
                global_orient = global_orient.reshape(1, -1).to(device)
                left_hand_pose = params['left_hand_pose'].reshape(1, -1).to(device) if 'left_hand_pose' in params else None
                right_hand_pose = params['right_hand_pose'].reshape(1, -1).to(device) if 'right_hand_pose' in params else None
                smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
            elif body_model_name == 'smpl':
                body_pose = params['body_pose'][None, :, :, :] # (1, 23, 3, 3)
                global_orient = params['global_orient'][None, :, :] # (1, 1, 3, 3)
                betas = params['betas'][None, :] # (1, 10)
                smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, pose2rot=False)
            else:
                raise ValueError(f"Unknown body model: {body_model_name}")

            # Extract main body joints and visualize 3D skeleton from SMPL-X
            smplx_joints = smplx_output['joints']
            # Save the root joint (pelvis) translation for later compensation
            params['org_cam_root_transl'] = smplx_joints[0, 0,:3] #.to(device)

            if body_model_name == 'smplx':
                smplx_coco_main_body_joints = smplx_joints[0, smplx_main_body_joint_idx, :] #.to(device)
            elif body_model_name == 'smpl':
                smplx_coco_main_body_joints = smplx_joints[0, smpl_main_body_joint_idx, :] #.to(device)
            vitpose_2d_keypoints = multiview_multiperson_pose2d[human_name][cam_name][coco_main_body_joint_idx] # .cpu().numpy() # (J, 2+1)
            init_trans = estimate_initial_trans(smplx_coco_main_body_joints, vitpose_2d_keypoints, focal_length, princpt, COCO_MAIN_BODY_SKELETON)
            # How to use init_trans?
            # vertices = vertices - joints[0:1] + init_trans # !ALWAYS! 
            if human_name in sorted(list(first_cam_human_name_counts.keys())):
                first_cam_human_name_counts[human_name]['count'] += 1
                first_cam_human_name_counts[human_name]['pose2d_conf'] = sum(vitpose_2d_keypoints[:, 2])
            else:
                missing_human_names_in_first_cam[human_name].append(cam_name)
            multiview_multiperson_init_trans[human_name][cam_name] = init_trans

    # main human is the one that is detected in the first camera and has the most detections across all cameras
    main_human_name_candidates = []
    max_count = 0
    for human_name, count_dict in first_cam_human_name_counts.items():
        if count_dict['count'] == len(camera_names):
            main_human_name_candidates.append(human_name)
            max_count = len(camera_names)
        elif count_dict['count'] > max_count:
            max_count = count_dict['count']
            main_human_name_candidates.append(human_name)
    
    if max_count != len(camera_names):
        print(f"Warning: {main_human_name_candidates} are the most detected main human but not detected in all cameras")

    # First filter to only keep humans with the maximum count
    max_count_humans = []
    for human_name in main_human_name_candidates:
        if first_cam_human_name_counts[human_name]['count'] == max_count:
            max_count_humans.append(human_name)
    
    # Among those with max count, pick the one with highest confidence
    main_human_name = None
    max_conf = 0
    for human_name in max_count_humans:
        conf = first_cam_human_name_counts[human_name]['pose2d_conf']
        if conf > max_conf:
            max_conf = conf
            main_human_name = human_name
    print("The main (reference) human name for the scale initilization is: ", main_human_name)

    # Initialize Stage 2: Get the initial camera poses with respect to the first camera
    global_orient_first_cam = multiview_multiple_human_cam_pred[first_cam][main_human_name]['params']['global_orient'][0] # .to(device)
    # Convert axis angle to rotation matrix using torch
    if body_model_name == 'smplx':
        global_orient_first_cam = torch_angle_axis_to_rotation_matrix(global_orient_first_cam).reshape(3,3)
    
    init_trans_first_cam = multiview_multiperson_init_trans[main_human_name][first_cam]

    # First camera (world coordinate) pose
    world_T_first = torch.eye(4, device=device)

    # Calculate other camera poses relative to world (first camera)
    cam_poses = {first_cam: world_T_first}
    for cam_name in sorted(list(multiview_multiperson_init_trans[main_human_name].keys())):
        if cam_name == first_cam:
            continue
        
        # Get human orientation and position in other camera
        global_orient_other_cam = multiview_multiple_human_cam_pred[cam_name][main_human_name]['params']['global_orient'][0] #.to(device)
        if body_model_name == 'smplx':
            global_orient_other_cam = torch_angle_axis_to_rotation_matrix(global_orient_other_cam).reshape(3,3)
        init_trans_other_cam = multiview_multiperson_init_trans[main_human_name][cam_name]

        # Calculate rotation and translation
        R_other = torch.matmul(global_orient_first_cam, global_orient_other_cam.transpose(0,1))
        t_other = init_trans_first_cam - torch.matmul(R_other, init_trans_other_cam)

        # Create 4x4 transformation matrix
        T_other = torch.eye(4, device=device)
        T_other[:3, :3] = R_other
        T_other[:3, 3] = t_other

        cam_poses[cam_name] = T_other

    # Visualize the camera poses (cam to world (first cam))
    # visualize_cameras(cam_poses)

    # Now cam_poses contains all camera poses in world coordinates
    # The poses can be used to initialize the scene

    # Organize the data for optimization
    # Get the first cam human parameters with the initial translation
    first_cam_human_params = {}
    for human_name in sorted(list(multiview_multiple_human_cam_pred[first_cam].keys())):
        first_cam_human_params[human_name] = multiview_multiple_human_cam_pred[first_cam][human_name]['params']
        first_cam_human_params[human_name]['root_transl'] = multiview_multiperson_init_trans[human_name][first_cam].reshape(1, -1) #.to(device)

    # Initialize Stage 3: If the first camera (world coordinate frame) has missing person,
    # move other camera view's human to the first camera view's human's location
    if True:
        # TODO: convert the operations to torch
        try:
            for missing_human_name in missing_human_names_in_first_cam:
                missing_human_exist_cam_idx = 0
                other_cam_name = missing_human_names_in_first_cam[missing_human_name][missing_human_exist_cam_idx]
                while other_cam_name not in sorted(list(cam_poses.keys())):
                    missing_human_exist_cam_idx += 1
                    if missing_human_exist_cam_idx == len(missing_human_names_in_first_cam[missing_human_name]):
                        print(f"Warning: {missing_human_name} cannot be handled because it can't transform to the first camera coordinate frame")
                        continue
                    other_cam_name = missing_human_names_in_first_cam[missing_human_name][missing_human_exist_cam_idx]
                missing_human_params_in_other_cam = multiview_multiple_human_cam_pred[other_cam_name][missing_human_name]['params']
                # keys: 'body_pose', 'betas', 'global_orient', 'right_hand_pose', 'left_hand_pose', 'transl'
                # transform the missing_human_params_in_other_cam to the first camera coordinate frame
                other_cam_to_first_cam_transformation = cam_poses[other_cam_name] # (4,4)
                missing_human_params_in_other_cam_global_orient = missing_human_params_in_other_cam['global_orient'][0].cpu().numpy() # (3,)
                missing_human_params_in_other_cam_global_orient = R.from_rotvec(missing_human_params_in_other_cam_global_orient).as_matrix().astype(np.float32) # (3,3)
                missing_human_params_in_other_cam_global_orient = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_params_in_other_cam_global_orient # (3,3)
                missing_human_params_in_other_cam['global_orient'] = torch.from_numpy(R.from_matrix(missing_human_params_in_other_cam_global_orient).as_rotvec().astype(np.float32)).to(device) # (3,)

                missing_human_init_trans_in_other_cam = multiview_multiperson_init_trans[missing_human_name][other_cam_name]
                missing_human_init_trans_in_first_cam = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_init_trans_in_other_cam + other_cam_to_first_cam_transformation[:3, 3]
                # compenstate rotation (translation from origin to root joint was not cancled)
                root_transl_compensator = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_params_in_other_cam['org_cam_root_transl'] 
                missing_human_init_trans_in_first_cam = missing_human_init_trans_in_first_cam + root_transl_compensator
                #
                missing_human_params_in_other_cam['root_transl'] = torch.from_numpy(missing_human_init_trans_in_first_cam).reshape(1, -1).to(device)

                first_cam_human_params[missing_human_name] = missing_human_params_in_other_cam
        except:
            print("Warning: Some humans are missing in the first camera view and cannot be handled")
        
    # Visualize the first cam human parameters with the camera poses
    # decode the human parameters to 3D vertices and visualize
    if get_vertices:
        first_cam_human_vertices = {}
        for human_name, human_params in first_cam_human_params.items():
            body_pose = human_params['body_pose'] #.reshape(1, -1).to(device)
            global_orient = human_params['global_orient'] #.reshape(1, -1).to(device)
            betas = human_params['betas'] #.reshape(1, -1).to(device)
            if body_model_name == 'smplx':
                body_pose = body_pose.reshape(1, -1).to(device)
                betas = betas.reshape(1, -1).to(device)
                global_orient = global_orient.reshape(1, -1).to(device)
                left_hand_pose = human_params['left_hand_pose'].reshape(1, -1).to(device) if 'left_hand_pose' in human_params else None
                right_hand_pose = human_params['right_hand_pose'].reshape(1, -1).to(device) if 'right_hand_pose' in human_params else None
                smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
            elif body_model_name == 'smpl':
                body_pose = human_params['body_pose'][None, :, :, :] # (1, 23, 3, 3)
                global_orient = human_params['global_orient'][None, :, :] # (1, 1, 3, 3)
                betas = human_params['betas'][None, :] # (1, 10)
                smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, pose2rot=False)
            else:
                raise ValueError(f"Unknown body model: {body_model_name}")

            vertices = smplx_output.vertices[0].detach().cpu().numpy()
            joints = smplx_output.joints[0].detach().cpu().numpy()
            vertices = vertices - joints[0:1] + human_params['root_transl'].detach().cpu().numpy()
            first_cam_human_vertices[human_name] = vertices
            # visualize_cameras_and_human(cam_poses, human_vertices=first_cam_human_vertices, smplx_faces=smplx_layer.faces)
    else:
        first_cam_human_vertices = None

    optim_target_dict = {} # human_name: str -> Dict[param_name: str -> nn.Parameter]
    for human_name, human_params in first_cam_human_params.items():
        optim_target_dict[human_name] = {}
        
        # Convert human parameters to nn.Parameters for optimization
        # matrix to 6d
        if body_model_name == 'smpl':
            global_orient_6d = matrix_to_rotation_6d(human_params['global_orient'])
            optim_target_dict[human_name]['global_orient'] = nn.Parameter(global_orient_6d).float().to(device)
            body_pose_6d = matrix_to_rotation_6d(human_params['body_pose'])
            optim_target_dict[human_name]['body_pose'] = nn.Parameter(body_pose_6d).float().to(device)
        else:
            optim_target_dict[human_name]['global_orient'] = nn.Parameter(human_params['global_orient'].float().to(device))
            optim_target_dict[human_name]['body_pose'] = nn.Parameter(human_params['body_pose'].float().to(device))

        optim_target_dict[human_name]['betas'] = nn.Parameter(human_params['betas'].float().to(device))
        optim_target_dict[human_name]['root_transl'] = nn.Parameter(human_params['root_transl'].float().to(device)) # (1, 3)

        if body_model_name == 'smplx':
            for param_name in optim_target_dict[human_name].keys():
                # reshape to (1, -1)
                optim_target_dict[human_name][param_name] = nn.Parameter(human_params[param_name].reshape(1, -1).float().to(device))
            if 'left_hand_pose' in human_params:
                optim_target_dict[human_name]['left_hand_pose'] = nn.Parameter(human_params['left_hand_pose'].float().to(device))  # (1, 45)
            else:
                optim_target_dict[human_name]['left_hand_pose'] = nn.Parameter(torch.zeros(1, 45).float().to(device))  # (1, 45)
            if 'right_hand_pose' in human_params:
                optim_target_dict[human_name]['right_hand_pose'] = nn.Parameter(human_params['right_hand_pose'].float().to(device))  # (1, 45)
            else:
                optim_target_dict[human_name]['right_hand_pose'] = nn.Parameter(torch.zeros(1, 45).float().to(device))  # (1, 45)

    return optim_target_dict, cam_poses, first_cam_human_vertices

def get_human_loss(smplx_layer_dict, body_model_name, humans_optim_target_dict, cam_names, multiview_world2cam_4by4, multiview_intrinsics, multiview_multiperson_poses2d, multiview_multiperson_bboxes, only_main_body_joints=True, shape_prior_weight=0, device='cuda'):
    # multiview_multiperson_poses2d: Dict[human_name -> Dict[cam_name -> (J, 3)]]
    # multiview_multiperson_bboxes: Dict[human_name -> Dict[cam_name -> (5)]]
    # multiview_world2cam_4by4: (N, 4, 4), multiview_intrinsics: (N, 3, 3)

    # save the 2D joints for visualization
    projected_joints = defaultdict(dict)

    # Collect all human parameters into batched tensors
    human_names = sorted(list(humans_optim_target_dict.keys()))
    num_of_humans_for_optimization = batch_size = len(human_names)

    # # define the smplx layer
    smplx_layer = smplx_layer_dict[batch_size]

    # Batch all SMPL parameters
    # change 6d rotation to rotation matrix
    if body_model_name == 'smplx':
        body_pose = torch.cat([humans_optim_target_dict[name]['body_pose'].reshape(1, -1) for name in human_names], dim=0)
        betas = torch.cat([humans_optim_target_dict[name]['betas'].reshape(1, -1) for name in human_names], dim=0)
        global_orient = torch.cat([humans_optim_target_dict[name]['global_orient'].reshape(1, -1) for name in human_names], dim=0)
        root_transl = torch.cat([humans_optim_target_dict[name]['root_transl'].reshape(1, 1, -1) for name in human_names], dim=0)
        left_hand_pose = torch.cat([humans_optim_target_dict[name]['left_hand_pose'].reshape(1, -1) for name in human_names], dim=0)
        right_hand_pose = torch.cat([humans_optim_target_dict[name]['right_hand_pose'].reshape(1, -1) for name in human_names], dim=0)

        # Forward pass through SMPL-X model for all humans at once
        smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
    
    elif body_model_name == 'smpl':
        body_pose = torch.cat([humans_optim_target_dict[name]['body_pose'].reshape(1, 23, 6) for name in human_names], dim=0)
        betas = torch.cat([humans_optim_target_dict[name]['betas'].reshape(1, 10) for name in human_names], dim=0)
        global_orient = torch.cat([humans_optim_target_dict[name]['global_orient'].reshape(1, 1, 6) for name in human_names], dim=0)
        root_transl = torch.cat([humans_optim_target_dict[name]['root_transl'].reshape(1, 1, 3) for name in human_names], dim=0)

        # change 6d rotation to rotation matrix
        global_orient = rotation_6d_to_matrix(global_orient)
        body_pose = rotation_6d_to_matrix(body_pose)
        # Forward pass through SMPL model for all humans at once
        smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, pose2rot=False)
    
    # Add root translation to joints
    smplx_j3d = smplx_output.joints  # (B, J, 3)
    smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl  # (B, J, 3)

    # Project joints to all camera views at once
    # Reshape for batch projection
    B, J, _ = smplx_j3d.shape
    N = len(cam_names)
    
    # Expand camera parameters to match batch size
    world2cam_expanded = multiview_world2cam_4by4.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, 4, 4)
    intrinsics_expanded = multiview_intrinsics.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, 3, 3)
    
    # Expand joints to match number of cameras
    smplx_j3d_expanded = smplx_j3d.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, J, 3)
    
    # Project all joints at once
    if num_of_humans_for_optimization == batch_size:
        points_homo = torch.cat((smplx_j3d_expanded, torch.ones((B, N, J, 1), device=device)), dim=3)  # (B, N, J, 4)
        points_cam = torch.matmul(world2cam_expanded, points_homo.transpose(2, 3))  # (B, N, 4, J)
        points_img = torch.matmul(intrinsics_expanded, points_cam[:, :, :3, :])  # (B, N, 3, J)
        points_img = points_img[:, :, :2, :] / points_img[:, :, 2:3, :]  # (B, N, 2, J)
        points_img = points_img.transpose(2, 3)  # (B, N, J, 2)
    else:
        # if num_of_humans_for_optimization < 1:
        #     raise ValueError(f"num_of_humans_for_optimization must be greater than 0, but got {num_of_humans_for_optimization}")
        points_homo = torch.cat((smplx_j3d_expanded, torch.ones((B, N, J, 1), device=device)), dim=3)  # (B, N, J, 4)
        # detach the world2cam_expanded and intrinsics_expanded for the elements from num_of_humans_for_optimization to batch_size
        world2cam_expanded_detached = world2cam_expanded[num_of_humans_for_optimization:].detach()
        intrinsics_expanded_detached = intrinsics_expanded[num_of_humans_for_optimization:].detach()
        world2cam_expanded = torch.cat((world2cam_expanded[:num_of_humans_for_optimization], world2cam_expanded_detached), dim=0)
        intrinsics_expanded = torch.cat((intrinsics_expanded[:num_of_humans_for_optimization], intrinsics_expanded_detached), dim=0)
        points_cam = torch.matmul(world2cam_expanded, points_homo.transpose(2, 3))  # (B, N, 4, J)
        points_img = torch.matmul(intrinsics_expanded, points_cam[:, :, :3, :])  # (B, N, 3, J)
        points_img = points_img[:, :, :2, :] / points_img[:, :, 2:3, :]  # (B, N, 2, J)
        points_img = points_img.transpose(2, 3)  # (B, N, J, 2)

    # Initialize total loss
    total_loss = 0

    # Process each human's loss in parallel
    for human_idx, human_name in enumerate(human_names):
        # Get camera indices and loss weights for this human
        cam_indices = []
        loss_weights = []
        poses2d = []
        bbox_areas = 0
        
        for cam_name, bbox in multiview_multiperson_bboxes[human_name].items():
            bbox_area = bbox[2] * bbox[3]
            det_score = bbox[4]
            loss_weights.append(det_score / bbox_area)
            # bbox_areas += bbox_area # Don't need this for now
            cam_indices.append(cam_names.index(cam_name))
            poses2d.append(multiview_multiperson_poses2d[human_name][cam_name])

        loss_weights = torch.stack(loss_weights).float().to(device)
        poses2d = torch.stack(poses2d).float().to(device)  # (num_cams, J, 3)

        # Get projected joints for this human
        human_proj_joints = points_img[human_idx, cam_indices]  # (num_cams, J, 2)

        # Create COCO ordered joints
        human_proj_joints_coco = torch.zeros(len(cam_indices), len(COCO_WHOLEBODY_KEYPOINTS), 3, device=device, dtype=torch.float32)
        for i, joint_name in enumerate(COCO_WHOLEBODY_KEYPOINTS):

            if body_model_name == 'smplx' and joint_name in ORIGINAL_SMPLX_JOINT_NAMES:
                human_proj_joints_coco[:, i, :2] = human_proj_joints[:, ORIGINAL_SMPLX_JOINT_NAMES.index(joint_name), :2]
                human_proj_joints_coco[:, i, 2] = 1
            elif body_model_name == 'smpl' and joint_name in SMPL_45_KEYPOINTS:
                human_proj_joints_coco[:, i, :2] = human_proj_joints[:, SMPL_45_KEYPOINTS.index(joint_name), :2]
                human_proj_joints_coco[:, i, 2] = 1

        # Weight main body joints more heavily
        human_proj_joints_coco[:, :COCO_WHOLEBODY_KEYPOINTS.index('right_heel')+1, 2] *= 10

        # Get only main body keypoints
        if only_main_body_joints:
            human_proj_joints_coco = human_proj_joints_coco[:, coco_main_body_joint_idx, :]
            poses2d = poses2d[:, coco_main_body_joint_idx, :]

        # Compute MSE loss with weights 
        one_human_loss = loss_weights[:, None, None].repeat(1, human_proj_joints_coco.shape[1], 1) \
            * human_proj_joints_coco[:, :, 2:] * poses2d[:, :, 2:] \
            * F.mse_loss(human_proj_joints_coco[:, :, :2], poses2d[:, :, :2], reduction='none').mean(dim=-1, keepdim=True)

        total_loss += one_human_loss.mean()

        # Store projected joints for visualization
        for idx, cam_idx in enumerate(cam_indices):
            projected_joints[cam_names[cam_idx]][human_name] = human_proj_joints_coco[idx]
        
    # Add shape prior if requested
    if shape_prior_weight > 0:
        total_loss += shape_prior_weight * F.mse_loss(betas, torch.zeros_like(betas))

    return total_loss, projected_joints

def get_stage_optimizer(human_params, body_model_name, scene_params, residual_scene_scale, stage: int, lr: float = 0.01):
    # 1st stage; optimize the scene scale, human root translation, shape (beta), and global orientation parameters
    # 2nd stage; optimize the dust3r scene parameters +  human root translation, shape (beta), and global orientation
    # 3rd stage; 2nd stage + human local poses
    # human param names: ['root_transl', 'betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose']

    if stage == 1: # 1st
        optimizing_param_names = ['root_transl', 'betas'] # , 'global_orient'

        human_params_to_optimize = []
        human_params_names_to_optimize = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in sorted(list(optim_target_dict.keys())):
                if param_name in optimizing_param_names:
                    optim_target_dict[param_name].requires_grad = True    
                    human_params_to_optimize.append(optim_target_dict[param_name])
                    human_params_names_to_optimize.append(f'{human_name}_{param_name}')
                else:
                    optim_target_dict[param_name].requires_grad = False

        optimizing_params = human_params_to_optimize + [residual_scene_scale]

    elif stage == 2: # 2nd
        optimizing_human_param_names = ['root_transl', 'betas', 'global_orient']

        human_params_to_optimize = []
        human_params_names_to_optimize = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in sorted(list(optim_target_dict.keys())):
                if param_name in optimizing_human_param_names:
                    optim_target_dict[param_name].requires_grad = True
                    human_params_to_optimize.append(optim_target_dict[param_name])
                    human_params_names_to_optimize.append(f'{human_name}_{param_name}')
                else:
                    optim_target_dict[param_name].requires_grad = False

        optimizing_params = scene_params + human_params_to_optimize  

    elif stage == 3: # 3rd
        if body_model_name == 'smpl':
            optimizing_human_param_names = ['root_transl', 'betas', 'global_orient', 'body_pose']
        elif body_model_name == 'smplx':
            optimizing_human_param_names = ['root_transl', 'betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose']

        human_params_to_optimize = []
        human_params_names_to_optimize = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in sorted(list(optim_target_dict.keys())):
                if param_name in optimizing_human_param_names:
                    optim_target_dict[param_name].requires_grad = True
                    human_params_to_optimize.append(optim_target_dict[param_name])
                    human_params_names_to_optimize.append(f'{human_name}_{param_name}')
                else:
                    optim_target_dict[param_name].requires_grad = False

        optimizing_params = scene_params + human_params_to_optimize
    # Print optimization parameters
    print(f"Optimizing {len(optimizing_params)} parameters:")
    print(f"- Human parameters ({len(human_params_names_to_optimize)}): {human_params_names_to_optimize}")
    if stage == 2 or stage == 3:
        print(f"- Scene parameters ({len(scene_params)})")
    if stage == 1:
        print(f"- Residual scene scale (1)")
    optimizer = torch.optim.Adam(optimizing_params, lr=lr, betas=(0.9, 0.9))
    return optimizer

    
def convert_human_params_to_numpy(human_params):
    # convert human_params to numpy arrays and save to new dictionary
    human_params_np = {}
    for human_name, optim_target_dict in human_params.items():
        human_params_np[human_name] = {}
        for param_name in optim_target_dict.keys():
            human_params_np[human_name][param_name] = optim_target_dict[param_name].reshape(1, -1).detach().cpu().numpy()

    return human_params_np


def main(
        world_env_path: str, 
        bbox_dir: str='./tmp_demo_output/json_data', 
        pose2d_dir: str='./tmp_demo_output/pose2d', 
        smplx_dir: str='./tmp_demo_output/smplx', 
        out_dir: str = './tmp_demo_output', 
        person_ids: List[int] = [1, ],
        body_model_name: str = 'smplx',
        vis: bool = False
    ):
    """
    world_env_path: path to the world environment from Dust3r
    bbox_dir: directory containing the bounding box predictions from Dust3r
    pose2d_dir: directory containing the 2D pose predictions from Dust3r
    smpl_dir: directory containing the SMPL model from Dust3r
    out_dir: directory to save the aligned results after optimization
    person_ids: list of person ids to optimize; ex) --person-ids 1 2
    vis: whether to visualize the results
    """
    # I abuse 'cam_names' and 'frame_names' interchangeably - Hongsuk

    # Parameters I am tuning
    human_loss_weight = 5.0
    stage2_start_idx_percentage = 0.4 # 0.5 #0.2
    stage3_start_idx_percentage = 0.8
    min_niter = 500 # minimum number of optimization iterations
    max_niter = 2000 # maximum number of optimization iterations
    niter_factor = 15  # decides the length of optimization 
    update_scale_factor = 1.2 # if the first initialization is not good and the code is applying scale update with this factor
    min_scene_scale = 15 # minimum dust3r to metric scale factor
    lr = 0.015
    pose2d_conf_threshold = 0.5

    # Dust3r Config for the global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer #if num_of_cams > 2 else GlobalAlignerMode.PairViewer
    device = 'cuda'
    silent = False
    schedule = 'linear'
    lr_base = lr
    lr_min = 0.0001
    init = 'known_params_hongsuk'
    niter_PnP = 10
    min_conf_thr_for_pnp = 3
    norm_pw_scale = False
    shape_prior_weight = 1.0
    focal_break = 20 

    # Logistics 
    save_2d_pose_vis = 20 
    scene_loss_timer = Timer()
    human_loss_timer = Timer()
    gradient_timer = Timer()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vis_output_path = osp.join(out_dir, 'vis')
    Path(vis_output_path).mkdir(parents=True, exist_ok=True)

    """ Load the initial data """
    print('\033[92m' + "Loading initial data..." + '\033[0m' + f" time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # load the initial world environment
    with open(world_env_path, "rb") as f:
        world_env = pickle.load(f)

    dust3r_network_output = world_env["dust3r_network_output"]
    dust3r_ga_output = world_env["dust3r_ga_output"]

    frame_names, pts3d, im_focals, im_poses, affine_matrix_list, rgbimg_list = get_dust3r_init_data(dust3r_ga_output, device)  
    frame_idx_list = [int(f.split('_')[-1]) for f in frame_names]
    print("Total number of frames:", len(frame_names))

    # load the initial smpl output estimated 
    smplx_params_dict = get_smplx_init_data(smplx_dir, frame_idx_list, body_model_name)

    # load the 2D pose predictions
    pose2d_params_dict = get_pose2d_init_data(pose2d_dir, frame_idx_list)

    # load the bounding box predictions
    bbox_params_dict = get_bbox_init_data(bbox_dir, frame_idx_list)

    # load the sam2 mask predictions
    # this is just for later visualization
    # not used for optimization
    sam2_dir = bbox_dir.replace('json_data', 'mask_data')
    sam2_mask_params_dict = get_mask_init_data(sam2_dir, frame_idx_list)
    sam2_mask_params_dict_transformed = {}
    for idx, frame_idx in enumerate(frame_idx_list):
        sam2_mask = sam2_mask_params_dict[frame_idx]
        affine_matrix = affine_matrix_list[idx]
        
        # Convert 2x3 affine matrix to 3x3 homogeneous form
        affine_matrix_homog = np.vstack([affine_matrix, [0, 0, 1]])
        # Get inverse transform
        affine_matrix_inv = np.linalg.inv(affine_matrix_homog)
        # Extract the 2x3 portion needed for cv2.warpAffine
        affine_matrix_inv_2x3 = affine_matrix_inv[:2, :]

        # Apply inverse transform using cv2.warpAffine
        sam2_mask_transformed = cv2.warpAffine(
            sam2_mask, 
            affine_matrix_inv_2x3, 
            (rgbimg_list[idx].shape[1], rgbimg_list[idx].shape[0]),
            flags=cv2.INTER_NEAREST
        )

        frame_name = frame_names[idx]
        sam2_mask_params_dict_transformed[frame_name] = sam2_mask_transformed

        # TEMP Vis
        # rgb = rgbimg_list[idx]
        # rgb[sam2_mask_transformed > 0] = 0
        # cv2.imwrite(osp.join(vis_output_path, f'frame_{frame_name}_sam2_mask.png'), rgb[..., ::-1] * 255)

    """ Rearrange the data for optimization """
    print('\033[92m' + "Rearranging the data for optimization..." + '\033[0m'+ f" time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    multiview_affine_transforms = {}
    multiview_images = {}
    for idx, frame_name in enumerate(frame_names):
        # get the affine matrix
        affine_matrix = affine_matrix_list[idx]
        multiview_affine_transforms[frame_name] = affine_matrix
        multiview_images[frame_name] = rgbimg_list[idx]


    # Put pose2d, bbox, and smpl parameters in the same dictionary
    multiview_multiple_human_cam_pred = {}
    for frame_idx, frame_name in zip(frame_idx_list, frame_names):
        multiview_multiple_human_cam_pred[frame_name] = {}
        for person_id in person_ids:
            multiview_multiple_human_cam_pred[frame_name][person_id] = {}
            multiview_multiple_human_cam_pred[frame_name][person_id]['pose2d'] = pose2d_params_dict[frame_idx][person_id]['keypoints'] # (133, 2+1), COCO_WHOLEBODY_KEYPOINTS order
            multiview_multiple_human_cam_pred[frame_name][person_id]['bbox'] = bbox_params_dict[frame_idx][person_id] # (4+1, )
            smplx_data =  smplx_params_dict[frame_idx][person_id] # Dict[str, torch.tensor]

            if body_model_name == 'smpl':
                smpl_params_tensor = {key: torch.tensor(smplx_data['smpl_params'][key], device=device, dtype=torch.float32) for key in smplx_data['smpl_params'].keys()}
                multiview_multiple_human_cam_pred[frame_name][person_id]['params'] = smpl_params_tensor
                # SMPL parameter shapes:
                # body_pose: (23, 3, 3), torch.tensor
                # betas: (10), torch.tensor
                # global_orient: (1, 3, 3), torch.tensor
            elif body_model_name == 'smplx':
                smplx_params_tensor = {key: torch.tensor(smplx_data[key], device=device, dtype=torch.float32) for key in smplx_data.keys() if smplx_data[key] is not None}
                multiview_multiple_human_cam_pred[frame_name][person_id]['params'] = smplx_params_tensor
                # SMPLX parameter shapes:
                # body_pose: (21, 3), torch.tensor
                # betas: (10), torch.tensor
                # global_orient: (1, 3), torch.tensor
                # right_hand_pose: (15, 3), torch.tensor
                # left_hand_pose: (15, 3), torch.tensor
            else:
                raise ValueError(f"Unknown body model: {body_model_name}")


    # Affine transform the pose2d and bbox to the Dust3r input image size
    multiview_multiperson_poses2d = defaultdict(dict)
    multiview_multiperson_bboxes = defaultdict(dict)

    # Process each camera and human
    for cam_name in sorted(list(multiview_multiple_human_cam_pred.keys())):
        for human_name in sorted(list(multiview_multiple_human_cam_pred[cam_name].keys())):
            # Get pose2d and bbox data
            pose2d = torch.tensor(multiview_multiple_human_cam_pred[cam_name][human_name]['pose2d'], device=device, dtype=torch.float32)
            bbox = torch.tensor(multiview_multiple_human_cam_pred[cam_name][human_name]['bbox'], device=device, dtype=torch.float32)

            # set confidence of joints to 0 if it's below threshold
            pose2d[:, 2] = pose2d[:, 2] * (pose2d[:, 2] > pose2d_conf_threshold)
            
            # Set confidence values of bbox; proxy for the confidence of the pose2d
            bbox_conf = pose2d[coco_main_body_joint_idx, 2].mean()
            bbox = torch.cat([bbox[:4], torch.tensor([bbox_conf], device=device, dtype=torch.float32)], dim=0)

            # Convert affine transform to torch tensor and add homogeneous row
            affine_transform = torch.tensor(multiview_affine_transforms[cam_name], device=device, dtype=torch.float32)
            homogeneous_row = torch.tensor([[0, 0, 1]], device=device, dtype=torch.float32)
            affine_transform = torch.cat([affine_transform, homogeneous_row], dim=0)
            # Get inverse transform
            affine_transform = torch.linalg.inv(affine_transform)

            # Transform pose2d
            pose2d_conf = pose2d[:, 2].clone()
            pose2d[:, 2] = 1.0
            pose2d_homogeneous = pose2d.T
            pose2d_transformed = torch.matmul(affine_transform, pose2d_homogeneous)
            pose2d_transformed = pose2d_transformed.T
            
            # Preserve original confidence values
            pose2d_transformed[:, 2] = pose2d_conf

            dust3r_img_h, dust3r_img_w = multiview_images[cam_name].shape[:2] # (512, 288)
            # sanity check; if there's any pose2d_transformed's xy is out of the image size, make such joints' confidence 0
            pose2d_transformed[:, 2] = pose2d_transformed[:, 2] * (pose2d_transformed[:, 0] < dust3r_img_w) * (pose2d_transformed[:, 1] < dust3r_img_h)
            pose2d_transformed[:, 2] = pose2d_transformed[:, 2] * (pose2d_transformed[:, 0] > 0) * (pose2d_transformed[:, 1] > 0)

            # Transform bbox
            bbox_reshaped = torch.tensor([[bbox[0], bbox[1], 1], 
                                        [bbox[2], bbox[3], 1]], device=device, dtype=torch.float32)
            bbox_transformed = torch.matmul(affine_transform, bbox_reshaped.T)
            bbox_transformed = bbox_transformed.T
            bbox[:4] = bbox_transformed[:, :2].reshape(-1)

            if vis:
                img = (multiview_images[cam_name] * 255).astype(np.uint8)
                # Convert back to numpy for visualization
                pose2d_np = pose2d_transformed.cpu().numpy()
                bbox_np = bbox.cpu().numpy()
                img = draw_2d_keypoints(img, pose2d_np)
                img = cv2.rectangle(img, 
                                  (int(bbox_np[0]), int(bbox_np[1])), 
                                  (int(bbox_np[2]), int(bbox_np[3])), 
                                  (0, 255, 0), 2)

                img = cv2.putText(img, str(human_name), (int(bbox_np[0]), int(bbox_np[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imwrite(osp.join(vis_output_path, 
                          f'target_frame_{cam_name}_{human_name}_2d_keypoints_bbox.png'), 
                          img[..., ::-1])

            # Store transformed results
            multiview_multiperson_poses2d[human_name][cam_name] = pose2d_transformed
            multiview_multiperson_bboxes[human_name][cam_name] = bbox

    """ Initialize the human parameters """
    print('\033[92m' + "Initialize the human parameters..." + '\033[0m'+ f" time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # intrinsics for human translation initialization
    if body_model_name == 'smplx':
        smplx_layer_dict = {
            1: smplx.create(model_path = './body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = 1).to(device),
            len(person_ids): smplx.create(model_path = './body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = len(person_ids)).to(device),
        }
    else:
        smplx_layer_dict = {
            1: smplx.create(model_path = './body_models', model_type = 'smpl', gender = 'neutral', num_betas = 10, batch_size = 1).to(device),
            len(person_ids): smplx.create(model_path = './body_models', model_type = 'smpl', gender = 'neutral', num_betas = 10, batch_size = len(person_ids)).to(device),
        }
    init_focal_length = im_focals[0] 
    init_princpt = [256., 144.] 

    human_params, human_inited_cam_poses, first_cam_human_vertices = \
            init_human_params(smplx_layer_dict[1], body_model_name, multiview_multiple_human_cam_pred, multiview_multiperson_poses2d, init_focal_length, init_princpt, device, get_vertices=vis) # dict of human parameters
    init_human_cam_data = {
        'human_params': human_params,
        'human_inited_cam_poses': human_inited_cam_poses,
    }    

    """ Initialize the scene """
    print('\033[92m' + "Initialize the scene..." + '\033[0m'+ f" time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Convert camera locations to torch tensors and move to device
    human_inited_cam_locations = []
    dust3r_cam_locations = []
    for frame_name in sorted(list(human_inited_cam_poses.keys())):
        human_inited_cam_locations.append(human_inited_cam_poses[frame_name][:3, 3])
        dust3r_cam_locations.append(im_poses[frame_names.index(frame_name)][:3, 3])
    human_inited_cam_locations = torch.stack(human_inited_cam_locations)  # (N, 3)
    dust3r_cam_locations = torch.stack(dust3r_cam_locations)  # (N, 3)

    try:
        if len(human_inited_cam_locations) > 2:
            # human_inited_cam_locations: (N, 3), known, first row is the origin, which means it is (0,0,0)
            # dust3r_cam_locations: (N, 3), known, first row is the origin, which means it is (0,0,0)
            # scene_scale: scalar, unknown
            # Solve least squares problem Ax=b to find optimal scale factor
            # where A is dust3r_cam_locations, x is scene_scale, b is human_inited_cam_locations
            # Reshape to match equation form
            A = dust3r_cam_locations.reshape(-1, 1)  # (3N, 1)
            b = human_inited_cam_locations.reshape(-1)  # (3N,)
            
            # Solve normal equation: (A^T A)x = A^T b
            ATA = A.T @ A  # (1,1)
            ATb = A.T @ b  # (1,)
            
            if ATA[0,0] > 1e-6:  # Check for numerical stability
                scene_scale = ATb[0] / ATA[0,0]  # Scalar solution
                scene_scale = abs(scene_scale)
            else:
                raise ValueError("Dust3r camera locations are too close to zero")
        elif len(human_inited_cam_locations) == 2:
            # get the ratio between the two distances

            hmr2_cam_dist = torch.norm(human_inited_cam_locations[0] - human_inited_cam_locations[1])
            dust3r_cam_dist = torch.norm(dust3r_cam_locations[0] - dust3r_cam_locations[1])
            # check dust3r failure
            if dust3r_cam_dist < 1e-3: # camera distance is too small
                raise ValueError(f"Maybe Dust3r failure; Dust3r camera distance is too small: {dust3r_cam_dist:.3f}")
            dist_ratio = hmr2_cam_dist / dust3r_cam_dist
            scene_scale = abs(dist_ratio)
        else:
            print("Not enough camera locations to perform Procrustes alignment or distance ratio calculation")
            scene_scale = 80.0
        niter = min(max(int(niter_factor * scene_scale), min_niter), max_niter)

        print(f"Dust3r to Human original scale ratio: {scene_scale}")
        print(f"Set the number of iterations to {niter}; {niter_factor} * {scene_scale}")
        print(f"Rescaled Dust3r to Human scale ratio: {scene_scale}")

        # do the optimization again with scaled 3D points and camera poses
        pts3d_scaled = [p * scene_scale for p in pts3d]
        pts3d = pts3d_scaled
        im_poses[:, :3, 3] = im_poses[:, :3, 3] * scene_scale

    except:
        # print("Error in Procrustes alignment or distance ratio calculation due to zero division...")
        # print(f"Skipping this sample {sample['sequence']}_{sample['frame']}_{''.join(cam_names)}...")
        # continue
        print("Error in Procrustes alignment or distance ratio calculation due to Dust3r or HMR2 failure...")
        print("Setting the scale to 80.0 and switch to HMR2 initialized camera poses")
        scene_scale = 80.0
        niter = min(max(int(niter_factor * scene_scale), min_niter), max_niter)

        # Switch dust3r camera poses to the hmr2 initialized camera poses
        for cam_name in sorted(list(human_inited_cam_poses.keys())):
            im_poses[frame_names.index(cam_name)] = torch.from_numpy(human_inited_cam_poses[cam_name]).to(device)

        # do the optimization again with scaled 3D points and camera poses
        pts3d_scaled = [p * scene_scale for p in pts3d]
        pts3d = pts3d_scaled
        # Don't scale the camera locations
        # im_poses[:, :3, 3] = im_poses[:, :3, 3] * scene_scale

    # define the scene class that will be optimized
    scene = global_aligner(dust3r_network_output, device=device, mode=mode, verbose=not silent, focal_break=focal_break)
    scene.norm_pw_scale = norm_pw_scale

    # initialize the scene parameters with the known poses or point clouds
    if len(frame_names) >= 2:
        if init == 'known_params_hongsuk':
            print(f"Using known params initialization; im_focals: {im_focals}")
            scene.init_from_known_params_hongsuk(im_focals=im_focals, im_poses=im_poses, pts3d=pts3d, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
            
            print("Known params init")
        else:
            raise ValueError(f"Unknown initialization method: {init}")
    scene_params = [p for p in scene.parameters() if p.requires_grad]

    # Visualize the initilization of 3D human and 3D world
    if vis and first_cam_human_vertices is not None:
        world_env = parse_to_save_data(scene, frame_names)
        try:
            show_env_human_in_viser(world_env=world_env, world_scale_factor=1., smplx_vertices_dict=first_cam_human_vertices, smplx_faces=smplx_layer_dict[1].faces)
        except:
            import pdb; pdb.set_trace()
            
    print("Do the final check for the scene scale")
    res_scale = 1.0
    multiview_cam2world_4by4  = scene.get_im_poses().detach()  # (len(cam_names), 4, 4)
    multiview_world2cam_4by4 = torch.inverse(multiview_cam2world_4by4) # (len(cam_names), 4, 4)

    root_transl_list = []
    for human_name in human_params.keys():
        root_transl = human_params[human_name]['root_transl'].detach()
        root_transl_list.append(root_transl)
    root_transl_list = torch.stack(root_transl_list) # (N, 1, 3)
    # Apply cam2world to the root_transl
    root_transl_cam = (multiview_world2cam_4by4[:, None, :3, :3] @ root_transl_list[None, :, :, :].transpose(2,3)).transpose(2,3) + multiview_world2cam_4by4[:, None, :3, 3:].transpose(2,3) # (len(cam_names), N, 1, 3)
    root_transl_cam = root_transl_cam.reshape(-1, 3) # (len(cam_names) * N, 3)
    # check if all z of root_transl_cam are positive
    scale_update_iter = 0
    max_scale_update_iter = 10
    while (not (root_transl_cam[:, 2] > 0).all() or scene_scale < min_scene_scale) and scale_update_iter < max_scale_update_iter:
        # print("Some of the root_transl_cam have negative z values;")
        res_scale = res_scale * update_scale_factor
        scene_scale = scene_scale * update_scale_factor
        niter = min(max(int(niter * update_scale_factor), min_niter), max_niter)

        print(f"Rescaling the scene scale to {res_scale:.3f}")

        # Apply cam2world to the root_transl
        root_transl_cam = (multiview_world2cam_4by4[:, None, :3, :3] @ root_transl_list[None, :, :, :].transpose(2,3)).transpose(2,3) \
                + res_scale * multiview_world2cam_4by4[:, None, :3, 3:].transpose(2,3) # (len(cam_names), N, 1, 3)
        root_transl_cam = root_transl_cam.reshape(-1, 3) # (len(cam_names) * N, 3)
        scale_update_iter += 1

    if scale_update_iter == max_scale_update_iter:
        print("Warning: Maximum number of scale update iterations reached")
        print(f"Set niter to {max_niter}")
        niter = max_niter
    else:
        print("All root_transl_cam have positive z values")
    print(f"Dust3r world is scaled by {scene_scale:.3f}")
    residual_scene_scale = nn.Parameter(torch.tensor(res_scale, requires_grad=True).to(device))
   
    # 1st stage; ex) stage 1 is from 0% to 30%
    stage1_iter = list(range(0, int(niter * stage2_start_idx_percentage)))
    # 2nd stage; ex) stage 2 is from 30% to 60%
    stage2_iter = list(range(int(niter * stage2_start_idx_percentage), int(niter * stage3_start_idx_percentage)))
    # 3rd stage; ex) stage 3 is from 60% to 100%
    stage3_iter = list(range(int(niter * stage3_start_idx_percentage), niter))
    # Given the number of iterations, run the optimizer while forwarding the scene with the current parameters to get the loss

    """ Start the optimization """
    print('\033[92m' + "Initializing optimizer..." + '\033[0m'+ f" time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('\033[94m' + f"Number of iterations: {niter}" + '\033[0m')
    print('\033[94m' + f"Stage 1: 0% to {stage2_start_idx_percentage*100}%" + '\033[0m')
    print('\033[94m' + f"Stage 2: {stage2_start_idx_percentage*100}% to {stage3_start_idx_percentage*100}%" + '\033[0m') 
    print('\033[94m' + f"Stage 3: {stage3_start_idx_percentage*100}% to 100%" + '\033[0m')
    with tqdm.tqdm(total=niter) as bar:
        while bar.n < bar.total:
            # Set optimizer
            if bar.n == stage1_iter[0]:
                optimizer = get_stage_optimizer(human_params, body_model_name, scene_params, residual_scene_scale, 1, lr)
                print("\n1st stage optimization starts at ", bar.n)
            elif bar.n == stage2_iter[0]:
                human_loss_weight *= 2.
                lr_base = lr = 0.01
                optimizer = get_stage_optimizer(human_params, body_model_name, scene_params, residual_scene_scale, 2, lr)
                print("\n2nd stage optimization starts at ", bar.n)
                # Reinitialize the scene
                print("Residual scene scale: ", residual_scene_scale.item())
                scene_intrinsics = scene.get_intrinsics().detach().cpu().numpy()
                im_focals = [intrinsic[0,0] for intrinsic in scene_intrinsics]
                im_poses = scene.get_im_poses().detach()
                im_poses[:, :3, 3] = im_poses[:, :3, 3] * residual_scene_scale.item()
                pts3d = scene.get_pts3d()
                pts3d_scaled = [p * residual_scene_scale.item() for p in pts3d]
                scene.init_from_known_params_hongsuk(im_focals=im_focals, im_poses=im_poses, pts3d=pts3d_scaled, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
                print("Known params init")    
                
                if False and vis:
                    # Visualize the initilization of 3D human and 3D world
                    world_env = parse_to_save_data(scene, frame_names)
                    show_optimization_results(world_env, body_model_name, human_params, smplx_layer_dict[1])

            elif bar.n == stage3_iter[0]:
                human_loss_weight *= 0.5

                optimizer = get_stage_optimizer(human_params, body_model_name, scene_params, residual_scene_scale, 3, lr)
                print("\n3rd stage optimization starts at ", bar.n)

                if False and vis:
                    # Visualize the initilization of 3D human and 3D world
                    world_env = parse_to_save_data(scene, frame_names)
                    show_optimization_results(world_env, body_model_name, human_params, smplx_layer_dict[1])

            lr = adjust_lr(bar.n, niter, lr_base, lr_min, optimizer, schedule)
            optimizer.zero_grad()

            # get extrinsincs and intrinsics from the scene
            multiview_cam2world_4by4  = scene.get_im_poses()  # (len(cam_names), 4, 4)

            if bar.n in stage1_iter:
                multiview_cam2world_4by4 = multiview_cam2world_4by4.detach()
                # Create a new tensor instead of modifying in place
                multiview_cam2world_3by4 = torch.cat([
                    multiview_cam2world_4by4[:, :3, :3],
                    (multiview_cam2world_4by4[:, :3, 3] * residual_scene_scale).unsqueeze(-1)
                ], dim=2)
                multiview_cam2world_4by4 = torch.cat([
                    multiview_cam2world_3by4,
                    multiview_cam2world_4by4[:, 3:4, :]
                ], dim=1)
                # What originally I was doing. even for stage 2 and 3
                multiview_world2cam_4by4 = torch.inverse(multiview_cam2world_4by4) # (len(cam_names), 4, 4)
                multiview_intrinsics = scene.get_intrinsics().detach() # (len(cam_names), 3, 3)

            else:
                multiview_world2cam_4by4 = torch.inverse(multiview_cam2world_4by4) # (len(cam_names), 4, 4)
                multiview_intrinsics = scene.get_intrinsics() # (len(cam_names), 3, 3)

            # Initialize losses dictionary
            losses = {}

            # Get human loss
            human_loss_timer.tic()
            only_main_body_joints = not (bar.n in stage3_iter)
            losses['human_loss'], projected_joints = get_human_loss(smplx_layer_dict, body_model_name, human_params, frame_names, 
                                                                    multiview_world2cam_4by4, multiview_intrinsics, 
                                                                    multiview_multiperson_poses2d, multiview_multiperson_bboxes, 
                                                                    only_main_body_joints=only_main_body_joints, shape_prior_weight=shape_prior_weight, device=device)
            losses['human_loss'] = human_loss_weight * losses['human_loss']
            human_loss_timer.toc()

            if (bar.n in stage2_iter or bar.n in stage3_iter):
                # Get scene loss
                scene_loss_timer.tic()
                losses['scene_loss'] = scene()
                scene_loss_timer.toc()

            # Compute total loss
            total_loss = sum(losses.values())

            gradient_timer.tic()
            total_loss.backward()
            optimizer.step()
            gradient_timer.toc()

            # Create loss string for progress bar
            loss_str = f'{lr=:g} '
            loss_str += ' '.join([f'{k}={v:g}' for k, v in losses.items()])
            loss_str += f' total_loss={total_loss:g}'
            bar.set_postfix_str(loss_str)
            bar.update()

            if vis and bar.n % save_2d_pose_vis == 0:
                for frame_name, human_joints in projected_joints.items():
                    img = scene.imgs[frame_names.index(frame_name)].copy() * 255.
                    img = img.astype(np.uint8)
                    for human_name, joints in human_joints.items():
                        # darw the human name
                        img = cv2.putText(img, str(human_name), (int(joints[0, 0]), int(joints[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        for idx, joint in enumerate(joints):
                            img = cv2.circle(img, (int(joint[0]), int(joint[1])), 1, (0, 255, 0), -1)
                            # draw the index
                            # img = cv2.putText(img, f"{idx}", (int(joint[0]), int(joint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imwrite(osp.join(vis_output_path, f'optim_vis_{frame_name}_{bar.n}.png'), img[:, :, ::-1])
    
    print("Final losses:", ' '.join([f'{k}={v.item():g}' for k, v in losses.items()]))
    print(f"Time taken: human_loss={human_loss_timer.total_time:g}s, scene_loss={scene_loss_timer.total_time:g}s, backward={gradient_timer.total_time:g}s")

    # Save output
    total_output = {}
    total_output['hsfm_places_cameras'] = parse_to_save_data(scene, frame_names)
    for frame_name in total_output['hsfm_places_cameras'].keys():
        total_output['hsfm_places_cameras'][frame_name]['sam2_mask'] = sam2_mask_params_dict_transformed[frame_name]
    
    if body_model_name == 'smpl':
        for person_id in human_params.keys():
            human_params[person_id]['global_orient'] = rotation_6d_to_matrix(human_params[person_id]['global_orient'])
            human_params[person_id]['body_pose'] = rotation_6d_to_matrix(human_params[person_id]['body_pose'])
    total_output['hsfm_people(smplx_params)'] = convert_human_params_to_numpy(human_params)
    total_output['dust3r_places_cameras'] = dust3r_ga_output
    total_output['hmr2_people_cameras'] = init_human_cam_data 

    output_name = f'hsfm_output_{body_model_name}'
    print("Saving to ", osp.join(out_dir, f'{output_name}.pkl'))
    with open(osp.join(out_dir, f'{output_name}.pkl'), 'wb') as f:
        pickle.dump(total_output, f)    
    
    if vis:
        show_optimization_results(total_output['hsfm_places_cameras'], body_model_name, human_params, smplx_layer_dict[1])


if __name__ == "__main__":
    tyro.cli(main)