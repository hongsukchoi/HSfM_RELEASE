# -*- coding: utf-8 -*-
# @Time    : 2024/10/14
# @Author  : wenshao
# @Project : WiLoR-mini
# @FileName: test_wilor_pipeline.py

# -*- coding: utf-8 -*-
# @Time    : 2025/02/12
# @Author  : Hongsuk Choi

"""
you need to install trimesh and pyrender if you want to render mesh
pip install trimesh
pip install pyrender
"""

import os
import tyro
import time
import glob
import json
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
import pyrender
import numpy as np
import torch
import cv2

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from joint_names import COCO_WHOLEBODY_KEYPOINTS

coco_wholebody_right_hand_joint_indices = list(range(COCO_WHOLEBODY_KEYPOINTS.index('right_hand_root'), COCO_WHOLEBODY_KEYPOINTS.index('right_pinky') + 1))
coco_wholebody_left_hand_joint_indices = list(range(COCO_WHOLEBODY_KEYPOINTS.index('left_hand_root'), COCO_WHOLEBODY_KEYPOINTS.index('left_pinky') + 1))
LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


def create_raymond_lights():
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


class Renderer:

    def __init__(self, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(),
                                   vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t=None,
            rot=None,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])
        if is_right:
            mesh_base_color = mesh_base_color[::-1]
        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle,
                                        is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)


def test_wilor_image_pipeline(img_path):


    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    # img_path = "assets/img.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for _ in range(20):
        t0 = time.time()
        outputs = pipe.predict(image)
        print(time.time() - t0)
    save_dir = "./tmp"
    os.makedirs(save_dir, exist_ok=True)
    renderer = Renderer(pipe.wilor_model.mano.faces)

    render_image = image.copy()
    render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0
    pred_keypoints_2d_all = []
    for i, out in enumerate(outputs):
        # out.keys(): 'hand_bbox', 'is_right', 'wilor_preds'
        # hand_bbox: (4,)
        # is_right: float 1.0 or 0.0
        # wilor_preds: dict; 'global_orient', 'hand_pose', 'betas', 'pred_cam', 'pred_keypoints_3d', 'pred_vertices', 'pred_cam_t_full', 'scaled_focal_length', 'pred_keypoints_2d']
        # out['wilor_preds']['hand_pose'].shape: (1, 15, 3)
        # out['wilor_preds']['global_orient'].shape: (1, 1, 3)
        # out['wilor_preds']['betas'].shape: (1, 10)
        verts = out["wilor_preds"]['pred_vertices'][0]
        is_right = out['is_right']
        cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
        scaled_focal_length = out["wilor_preds"]['scaled_focal_length']
        pred_keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"]
        pred_keypoints_2d_all.append(pred_keypoints_2d)
        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
        tmesh.export(os.path.join(save_dir, f'{os.path.basename(img_path)}_hand{i:02d}.obj'))
        cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                        is_right=is_right,
                                        **misc_args)

        # Overlay image
        render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

    render_image = (255 * render_image).astype(np.uint8)
    for pred_keypoints_2d in pred_keypoints_2d_all:
        for j in range(pred_keypoints_2d[0].shape[0]):
            color = (0, 0, 255)
            radius = 3
            x, y = pred_keypoints_2d[0][j]
            cv2.circle(render_image, (int(x), int(y)), radius, color, -1)
            cv2.putText(render_image, str(j), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), render_image)
    print(os.path.join(save_dir, os.path.basename(img_path)))


def compute_iou(bbox1, bbox2):
    # compute the iou between bbox1 and bbox2
    # bbox1 and bbox2 are numpy arrays of shape (4,)
    # bbox1: x1, y1, x2, y2
    # bbox2: x1, y1, x2, y2
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2) 
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Check if boxes overlap
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    # Calculate areas
    area_intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area_bbox1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_bbox2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_union = area_bbox1 + area_bbox2 - area_intersection

    # Calculate IoU
    iou = area_intersection / area_union
    return iou

def main(
    img_dir: str='./demo_data/input_images/arthur_tyler_pass_by_nov20/cam01', 
    output_dir: str='./demo_data/input_3d_meshes/arthur_tyler_pass_by_nov20/cam01',
    pose2d_dir: str='./demo_data/input_2d_poses/arthur_tyler_pass_by_nov20/cam01',
    person_ids: list = [1, ],
    hand_bbox_thr: float = 0.7,
    vis: bool = False,
):
    # Must run ViTPose first to get the hand bounding boxes and assign the correct person id!

    output_dir = os.path.join(output_dir, os.path.basename(img_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)

    # Get all demo images that end with .jpg or .png
    img_paths = [img for img in Path(img_dir).glob('*.jpg')]
    if len(img_paths) == 0:
        img_paths = [img for img in Path(img_dir).glob('*.png')]
    img_paths.sort()

    result_dict = defaultdict(dict)
    if vis:            
        renderer = Renderer(pipe.wilor_model.mano.faces)
        print("Renderer initialized")

    for img_path in tqdm(img_paths):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        # Retrieve the hand bbox from bbox_dir
        frame_idx = int(img_path.stem.split('_')[-1])
        pose2d_path = Path(pose2d_dir) / f'pose_{frame_idx:05d}.json'
        # if pose2d_path does not exist, continue
        if not pose2d_path.exists():
            continue

        with open(pose2d_path, 'r') as f:
            pose2d_data = json.load(f)

        hand_bbox_dict = {person_id: {'right_hand': None, 'left_hand': None} for person_id in person_ids}
        for person_id in person_ids:
            pose2d = np.array(pose2d_data[str(person_id)]['keypoints'], dtype=np.float32)

            left_hand_joints = pose2d[coco_wholebody_left_hand_joint_indices]
            right_hand_joints = pose2d[coco_wholebody_right_hand_joint_indices]


            if left_hand_joints[:, 2].mean() > hand_bbox_thr:
                # compute the bounding box from the hand joints
                x1 = np.min(left_hand_joints[:, 0])
                y1 = np.min(left_hand_joints[:, 1])
                x2 = np.max(left_hand_joints[:, 0])
                y2 = np.max(left_hand_joints[:, 1])
                left_hand_bbox = np.array([x1, y1, x2, y2])

                # Sanitize the bbox
                left_hand_bbox[0] = max(0, left_hand_bbox[0])
                left_hand_bbox[1] = max(0, left_hand_bbox[1])
                left_hand_bbox[2] = min(image.shape[1], left_hand_bbox[2])
                left_hand_bbox[3] = min(image.shape[0], left_hand_bbox[3])

                hand_bbox_dict[person_id]['left_hand'] = left_hand_bbox

            if right_hand_joints[:, 2].mean() > hand_bbox_thr:
                # compute the bounding box from the hand joints
                x1 = np.min(right_hand_joints[:, 0])
                y1 = np.min(right_hand_joints[:, 1])
                x2 = np.max(right_hand_joints[:, 0])
                y2 = np.max(right_hand_joints[:, 1])
                right_hand_bbox = np.array([x1, y1, x2, y2])

                # Sanitize the bbox
                right_hand_bbox[0] = max(0, right_hand_bbox[0])
                right_hand_bbox[1] = max(0, right_hand_bbox[1])
                right_hand_bbox[2] = min(image.shape[1], right_hand_bbox[2])
                right_hand_bbox[3] = min(image.shape[0], right_hand_bbox[3])

                hand_bbox_dict[person_id]['right_hand'] = right_hand_bbox


        is_rights = []
        bboxes = []
        person_id_list = []
        for person_id in person_ids:
            if hand_bbox_dict[person_id]['right_hand'] is not None:
                is_rights.append(1)
                bboxes.append(hand_bbox_dict[person_id]['right_hand'])
                person_id_list.append(person_id)
            if hand_bbox_dict[person_id]['left_hand'] is not None:
                is_rights.append(0)
                bboxes.append(hand_bbox_dict[person_id]['left_hand'])
                person_id_list.append(person_id)

        assert len(is_rights) == len(bboxes) == len(person_id_list)
        bboxes = np.array(bboxes, dtype=np.float32)

        outputs = pipe.predict_with_bboxes(image, bboxes=bboxes, is_rights=is_rights)


        assert len(outputs) == len(bboxes)

        result_dict[frame_idx] = {person_id: {'right_hand': None, 'left_hand': None} for person_id in person_ids}
        for out_idx, out in enumerate(outputs):
            # hand_bbox: (4,)
            # is_right: float 1.0 or 0.0
            # The outputs are already numpy arrays, so we don't need to convert them to numpy arrays; https://github.com/warmshao/WiLoR-mini/blob/a20fc482e68d17c0c8fa19c64f3f4544b6a310cf/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py#L201
            # wilor_preds: dict; 'global_orient', 'hand_pose', 'betas', 'pred_cam', 'pred_keypoints_3d', 'pred_vertices', 'pred_cam_t_full', 'scaled_focal_length', 'pred_keypoints_2d']
            # out['wilor_preds']['hand_pose'].shape: (1, 15, 3)
            # out['wilor_preds']['global_orient'].shape: (1, 1, 3)
            # out['wilor_preds']['betas'].shape: (1, 10)
            # out['wilor_preds']['pred_keypoints_2d'].shape: (1, 21, 3)
            hand_bbox = np.array(out['hand_bbox'], dtype=np.float32) # (4,), x1y1x2y2 list

            saving_output = {
                'hand_bbox': hand_bbox, # (4,)
                'global_orient': out['wilor_preds']['global_orient'][0], # (1, 3)
                'hand_pose': out['wilor_preds']['hand_pose'][0], # (15, 3)
                'betas': out['wilor_preds']['betas'][0], # (10,)
                'pred_keypoints_3d': out['wilor_preds']['pred_keypoints_3d'][0], # (21, 3)
                'pred_keypoints_2d': out['wilor_preds']['pred_keypoints_2d'][0], # (21, 3)
            }   
            person_id = person_id_list[out_idx]

            is_right_from_vitpose = is_rights[out_idx]
            is_right_from_wilor = out['is_right']
            if is_right_from_vitpose != is_right_from_wilor:
                print(f"Mismatch between vitpose and wilor for frame {frame_idx} person {person_id}")
                continue

            if is_right_from_vitpose:
                result_dict[frame_idx][person_id]['right_hand'] = saving_output
            else:
                result_dict[frame_idx][person_id]['left_hand'] = saving_output

        with open(os.path.join(output_dir, f'mano_{frame_idx:05d}.pkl'), 'wb') as f:
            pickle.dump(result_dict[frame_idx], f)
        print(f"Saved mano_{frame_idx:05d}.pkl to {output_dir}")


        if vis:
            vis_save_dir = os.path.join(output_dir, 'vis')
            os.makedirs(vis_save_dir, exist_ok=True)
            render_image = image.copy()
            render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0
            pred_keypoints_2d_all = []
            
            for i, out in enumerate(outputs):
                # out.keys(): 'hand_bbox', 'is_right', 'wilor_preds'
                # hand_bbox: (4,)
                # is_right: float 1.0 or 0.0
                # wilor_preds: dict; 'global_orient', 'hand_pose', 'betas', 'pred_cam', 'pred_keypoints_3d', 'pred_vertices', 'pred_cam_t_full', 'scaled_focal_length', 'pred_keypoints_2d']
                # out['wilor_preds']['hand_pose'].shape: (1, 15, 3)
                # out['wilor_preds']['global_orient'].shape: (1, 1, 3)
                # out['wilor_preds']['betas'].shape: (1, 10)
                # out['wilor_preds']['pred_keypoints_2d'].shape: (1, 21, 2)
                # out['wilor_preds']['pred_keypoints_3d'].shape: (1, 21, 3)
                verts = out["wilor_preds"]['pred_vertices'][0]
                is_right = out['is_right']
                cam_t = out["wilor_preds"]['pred_cam_t_full'][0]
                scaled_focal_length = out["wilor_preds"]['scaled_focal_length']
                pred_keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"]
                pred_keypoints_2d_all.append(pred_keypoints_2d)
                misc_args = dict(
                    mesh_base_color=LIGHT_PURPLE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_PURPLE, is_right=is_right)
                save_file_name = f'{frame_idx:05d}_wilor_hand_pose_{i:02d}.obj'
                tmesh.export(os.path.join(vis_save_dir, save_file_name))
                cam_view = renderer.render_rgba(verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]],
                                                is_right=is_right,
                                                **misc_args)

                # Overlay image
                render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            render_image = (255 * render_image).astype(np.uint8)
            for pred_keypoints_2d in pred_keypoints_2d_all:
                for j in range(pred_keypoints_2d[0].shape[0]):
                    color = (0, 0, 255)
                    radius = 3
                    x, y = pred_keypoints_2d[0][j]
                    cv2.circle(render_image, (int(x), int(y)), radius, color, -1)
            save_file_name = f'{frame_idx:05d}_wilor_hand_pose_vis.jpg'
            cv2.imwrite(os.path.join(vis_save_dir, save_file_name), render_image)
            print("Saved to ", os.path.join(vis_save_dir, save_file_name))
        
    print("Done!")

if __name__ == '__main__':
    # img_path = '/home/hongsuk/projects/human_in_world/demo_data/input_images/jason_hands_dec13/frame_00083.jpg' #'/home/hongsuk/projects/human_in_world/demo_data/input_3d_meshes/arthur_tyler_pass_by_nov20/cam01/vis/00061_wilor_hand_pose_vis.jpg'
    # test_wilor_image_pipeline(img_path)
    tyro.cli(main)