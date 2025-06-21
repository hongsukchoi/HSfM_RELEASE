import os
import os.path as osp
import copy
import time
import pickle
import tyro
import numpy as np
import smplx
import torch
import viser
import cv2

from scipy.spatial.transform import Rotation as R
from scipy.linalg import orthogonal_procrustes

# Read colors from colors.txt file
colors_path = osp.join(osp.dirname(__file__), 'colors.txt')
colors = []
with open(colors_path, 'r') as f:
    for line in f:
        # Convert each line of RGB values to a list of integers
        rgb = list(map(int, line.strip().split()))
        colors.append(rgb)
COLORS = np.array(colors)

def get_color(idx):
    return COLORS[idx % len(COLORS)]

def visualize_cameras_and_human(cam_poses, human_vertices, smplx_faces, world_colmap_pointcloud_xyz=None, world_colmap_pointcloud_rgb=None):
    cam_poses = copy.deepcopy(cam_poses)
    human_vertices = copy.deepcopy(human_vertices)
    # set viser
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    # get rotation matrix of 180 degrees around x axis
    rot_180 = np.eye(3)
    rot_180[1, 1] = -1
    rot_180[2, 2] = -1  

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    human_idx = 0
    for human_name in sorted(human_vertices.keys()):
        vertices = human_vertices[human_name] @ rot_180       
        server.scene.add_mesh_simple(
            f"/{human_name}_human/mesh",
            vertices=vertices,
            faces=smplx_faces,
            flat_shading=False,
            wireframe=False,
            color=get_color(human_idx),
        )
        human_idx += 1


    cam_handles = []
    for cam_name, cam_pose in cam_poses.items():
        # Visualize the camera
        camera = cam_pose.copy()
        camera[:3, :3] = rot_180 @ camera[:3, :3] 
        camera[:3, 3] = camera[:3, 3] @ rot_180
        
        # rotation matrix to quaternion
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        # xyzw to wxyz
        quat = np.concatenate([quat[3:], quat[:3]])
        # translation vector
        trans = camera[:3, 3]

        # add camera
        cam_handle = server.scene.add_frame(
            f"/cam_{cam_name}",
            wxyz=quat,
            position=trans,
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.04,
        )
        cam_handles.append(cam_handle)

    # Add scene structure pointcloud
    if world_colmap_pointcloud_xyz is not None:
        world_colmap_pointcloud_xyz = world_colmap_pointcloud_xyz @ rot_180
        server.scene.add_point_cloud(
        "/world_colmap_pointcloud",
        points=world_colmap_pointcloud_xyz,
        colors=world_colmap_pointcloud_rgb,
        point_size=0.01,
        point_shape='circle'
    )

    # add transform controls, initialize the location with the first two cameras
    control0 = server.scene.add_transform_controls(
        "/controls/0",
        position=cam_handles[0].position,
        scale=cam_handles[0].axes_length,
    )
    control1 = server.scene.add_transform_controls(
        "/controls/1",
        position=cam_handles[1].position,
        scale=cam_handles[1].axes_length,
    )
    distance_text = server.gui.add_text("Distance", initial_value="Distance: 0")

    def update_distance():
        distance = np.linalg.norm(control0.position - control1.position)
        distance_text.value = f"Distance: {distance:.2f}"

        server.scene.add_spline_catmull_rom(
            "/controls/line",
            np.stack([control0.position, control1.position], axis=0),
            color=(255, 0, 0),
        )

    control0.on_update(lambda _: update_distance())
    control1.on_update(lambda _: update_distance())
    
    start_time = time.time()
    while True:
        time.sleep(0.01)
        timing_handle.value = (time.time() - start_time) 


def show_env_human_in_viser(world_env: dict = None, world_env_pkl: str = '', world_scale_factor: float = 1.,  smplx_vertices_dict: dict = None, smplx_faces: np.ndarray = None, gt_cameras: dict = None):
    if world_env is None:
        # Load world environment data estimated by Mast3r
        with open(world_env_pkl, 'rb') as f:
            world_env = pickle.load(f)
    
    disable_vis_human_pointcloud = True
    for img_name in world_env.keys():
        if img_name == 'non_img_specific':
            continue
        world_env[img_name]['pts3d'] *= world_scale_factor
        world_env[img_name]['cam2world'][:3, 3] *= world_scale_factor
        if 'sam2_mask' in world_env[img_name].keys():
            disable_vis_human_pointcloud = False

    
    # set viser
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    needs_update: bool = True

    def set_stale(_) -> None:
        nonlocal needs_update
        needs_update = True

    with server.gui.add_folder("Output Analysis"):
        gui_show_human_pointcloud = server.gui.add_checkbox("Show Human Pointcloud", True)
        if disable_vis_human_pointcloud:
            gui_show_human_pointcloud.disabled = True

        gui_show_human_mesh = server.gui.add_checkbox("Show Human Mesh", True)

        gui_confidence_threshold = server.gui.add_number(
            "Point Confidence >=", initial_value=3.5, min=0.0, max=20.0
        )
        gui_confidence_threshold.on_update(set_stale)

        # NEW: depth-gradient threshold for point filtering
        gui_gradient_threshold = server.gui.add_number(
                "Depth Gradient <", initial_value=0.05, min=0.0, max=0.5, step=0.002
            )
        gui_gradient_threshold.on_update(set_stale)

    with server.gui.add_folder("Visualization"):
        # frustum gui elements
        gui_line_width = server.gui.add_slider(
            "Frustum Line Width", initial_value=2.0, step=0.01, min=0.0, max=20.0
        )

        @gui_line_width.on_update
        def _(_) -> None:
            for cam in camera_frustums:
                cam.line_width = gui_line_width.value

        gui_frustum_scale = server.gui.add_slider(
            "Frustum Scale", initial_value=0.3, step=0.001, min=0.01, max=20.0
        )

        @gui_frustum_scale.on_update
        def _(_) -> None:
            for cam in camera_frustums:
                cam.scale = gui_frustum_scale.value

        gui_frustum_ours_color = server.gui.add_rgb(
            "Frustum RGB (ours)", initial_value=(255, 127, 14)
        )
        @gui_frustum_ours_color.on_update
        def _(_) -> None:
            for cam in camera_frustums:
                cam.color = gui_frustum_ours_color.value

        gui_point_size = server.gui.add_slider(
            "Point Size", initial_value=0.01, step=0.0001, min=0.001, max=0.05
        )
        gui_point_white = server.gui.add_slider(
            "Point White", initial_value=0.0, step=0.0001, min=0.0, max=1.0
        )
        gui_point_white.on_update(set_stale)

        @gui_point_size.on_update
        def _(_) -> None:
            for pc in pointcloud_handles:
                pc.point_size = gui_point_size.value

        gui_point_shape = server.gui.add_dropdown(
            "Point Shape", options=("circle", "square"), initial_value="circle"
        )

        @gui_point_shape.on_update
        def _(_) -> None:
            for pc in pointcloud_handles:
                pc.point_ball_norm = (
                    2.0 if gui_point_shape.value == "circle" else np.inf
                )    

    # get rotation matrix of 180 degrees around x axis
    rot_180 = np.eye(3)
    rot_180[1, 1] = -1
    rot_180[2, 2] = -1

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    pointcloud_handles = []
    human_mesh_handles = []
    cam_handles = []
    camera_frustums = [] #list[viser.CameraFrustumHandle]()
    gt_cam_handles = []
    control_distance_measurement = None
    start_time = time.time()
    try:
        while True:
            time.sleep(0.1)
            timing_handle.value = (time.time() - start_time) 

            with server.atomic():
            # if True:
                for img_idx, img_name in enumerate(sorted(world_env.keys())):
                    if img_name == 'non_img_specific':
                        continue
                    # Visualize the pointcloud of environment
                    pts3d = world_env[img_name]['pts3d']
                    if pts3d.ndim == 3:
                        pts3d = pts3d.reshape(-1, 3)
                    # points = pts3d[world_env[img_name]['msk'].flatten()]
                    # colors = world_env[img_name]['rgbimg'][world_env[img_name]['msk']].reshape(-1, 3)
                    # no masking
                    points = pts3d
                    colors =world_env[img_name]['rgbimg'].reshape(-1, 3)

                    points = points @ rot_180

                    # Filter out points with confidence < threshold and high depth gradient.
                    mask = world_env[img_name]['conf'].flatten() >= gui_confidence_threshold.value

                    # --- Depth-gradient filtering (optional) ---
                    if 'depths' in world_env[img_name]:
                        depths = world_env[img_name]['depths']  # (H, W)
                        dy, dx = np.gradient(depths)
                        grad_mag = np.sqrt(dx ** 2 + dy ** 2)
                        grad_mask = grad_mag.flatten() < gui_gradient_threshold.value
                        mask = mask & grad_mask
                    # -----------------------------------------

                    # show human pointcloud
                    if not gui_show_human_pointcloud.value and 'sam2_mask' in world_env[img_name].keys():
                        sam2_mask = world_env[img_name]['sam2_mask'].flatten()
                        # dilate the sam2_mask
                        kernel = np.ones((5, 5), np.uint8)
                        sam2_mask = cv2.dilate(sam2_mask, kernel, iterations=1).flatten() > 0
                        mask = mask & (sam2_mask == 0)

                    points_filtered = points[mask]
                    colors_filtered = colors[mask]

                    pointcloud_handle = server.scene.add_point_cloud(
                        f"/ours/pointcloud_{img_name}",
                        points=points_filtered,
                        point_size=gui_point_size.value,
                        point_shape=gui_point_shape.value,
                        colors=(1.0 - gui_point_white.value) * colors_filtered
                            + gui_point_white.value,
                    )
                    pointcloud_handles.append(pointcloud_handle)

                    # Visualize the camera
                    camera = world_env[img_name]['cam2world'].copy()
                    camera[:3, :3] = rot_180 @ camera[:3, :3] 
                    camera[:3, 3] = camera[:3, 3] @ rot_180
                    
                    # rotation matrix to quaternion
                    quat = R.from_matrix(camera[:3, :3]).as_quat()
                    # xyzw to wxyz
                    quat = np.concatenate([quat[3:], quat[:3]])
                    # translation vector
                    trans = camera[:3, 3]

                    # add camera frustum
                    rgbimg = world_env[img_name]['rgbimg']
                    K = world_env[img_name]['intrinsic']
                    # fov_rad = 2 * np.arctan(intrinsics_K[0, 2] / intrinsics_K[0, 0])DA
                    assert K.shape == (3, 3)
                    vfov_rad = 2 * np.arctan(K[1, 2] / K[1, 1])
                    aspect = rgbimg.shape[1] / rgbimg.shape[0]

                    camera_frustm = server.scene.add_camera_frustum(
                        f"/ours/{img_name}",
                        vfov_rad,
                        aspect,
                        scale=gui_frustum_scale.value,
                        line_width=gui_line_width.value,
                        color=gui_frustum_ours_color.value,
                        wxyz=quat,
                        position=trans,
                        image=rgbimg,
                    )
                    camera_frustums.append(camera_frustm)

                if len(human_mesh_handles) == 0 and smplx_vertices_dict is not None:
                    for img_idx, img_name in enumerate(sorted(smplx_vertices_dict.keys())):
                        vertices = smplx_vertices_dict[img_name]
                        vertices = vertices @ rot_180       
                        human_mesh_handle = server.scene.add_mesh_simple(
                            f"/{img_name}_main_human/mesh",
                            vertices=vertices,
                            faces=smplx_faces,
                            flat_shading=False,
                            wireframe=False,
                            color=get_color(img_idx),
                        )
                        human_mesh_handles.append(human_mesh_handle)
                
                if gt_cameras is not None:
                    for img_name in gt_cameras.keys():
                        # Visualize the gt camera
                        # camera = gt_cameras[img_name]
                        # cam2world_Rt_homo = camera['cam2world_4by4'].copy()
                        cam2world_Rt_homo = gt_cameras[img_name]

                        cam2world_R = rot_180 @ cam2world_Rt_homo[:3, :3]
                        cam2world_t = cam2world_Rt_homo[:3, 3] @ rot_180

                        # rotation matrix to quaternion
                        quat = R.from_matrix(cam2world_R).as_quat()
                        # xyzw to wxyz
                        quat = np.concatenate([quat[3:], quat[:3]])
                        # translation vector
                        trans = cam2world_t   

                        # add camera
                        gt_cam_handle = server.scene.add_frame(
                            f"/gt_cam_{img_name}",
                            wxyz=quat,
                            position=trans,
                        )
                        gt_cam_handles.append(gt_cam_handle)

                if control_distance_measurement is not None:
                    control0.on_update(lambda _: update_distance())
                    control1.on_update(lambda _: update_distance())

                # show human mesh
                for human_mesh_handle in human_mesh_handles:
                    human_mesh_handle.visible = gui_show_human_mesh.value

                # add transform controls, initialize the location with the first two cameras
                if control_distance_measurement is None:
                    control0 = server.scene.add_transform_controls(
                        "/controls/0",
                        position=camera_frustums[0].position,
                        scale=0.5
                    )
                    control1 = server.scene.add_transform_controls(
                        "/controls/1",
                        position=camera_frustums[1].position,
                        scale=0.5
                    )
                    distance_text = server.gui.add_text("Distance", initial_value="Distance: 0")
                    control_distance_measurement = True

                    def update_distance():
                        distance = np.linalg.norm(control0.position - control1.position)
                        distance_text.value = f"Distance: {distance:.2f}"

                        server.scene.add_spline_catmull_rom(
                            "/controls/line",
                            np.stack([control0.position, control1.position], axis=0),
                            color=(255, 0, 0),  
                        )

    except KeyboardInterrupt:
        pass

def show_optimization_results(world_env, body_model_name, human_params, smplx_layer):
    smplx_vertices_dict = {}
    for human_name, optim_target_dict in human_params.items():
        if body_model_name == 'smplx':
            # extract data from the optim_target_dict
            body_pose = optim_target_dict['body_pose'].reshape(1, -1)
            betas = optim_target_dict['betas'].reshape(1, -1)
            global_orient = optim_target_dict['global_orient'].reshape(1, -1)
            left_hand_pose = optim_target_dict['left_hand_pose'].reshape(1, -1)
            right_hand_pose = optim_target_dict['right_hand_pose'].reshape(1, -1)

            # decode the smpl mesh and joints
            smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
        
        elif body_model_name == 'smpl':
            body_pose = optim_target_dict['body_pose'][None, :, :, :] # (1, 23, 3, 3)
            global_orient = optim_target_dict['global_orient'][None, :, :] # (1, 1, 3, 3)
            betas = optim_target_dict['betas'][None, :] # (1, 10)
            smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, pose2rot=False)
        
        else:
            raise ValueError(f"Unknown body model: {body_model_name}")

        # Add root translation to the joints
        root_transl = optim_target_dict['root_transl'].reshape(1, 1, -1)
        smplx_vertices = smplx_output.vertices
        smplx_j3d = smplx_output.joints # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters
        smplx_vertices = smplx_vertices - smplx_j3d[:, 0:1, :] + root_transl
        smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl # !ALWAYS! 
        smplx_vertices_dict[human_name] = smplx_vertices[0].detach().cpu().numpy()
    try:
        show_env_human_in_viser(world_env=world_env, world_scale_factor=1., smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.faces)
    except:
        import pdb; pdb.set_trace()

def main(hsfm_pkl: str, body_model_name: str = 'smplx'):
    device = 'cuda'
     
    if body_model_name == 'smplx':
        smplx_layer  = smplx.create(model_path = './body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = 1).to(device)
    else:
        smplx_layer  = smplx.create(model_path = './body_models', model_type = 'smpl', gender = 'neutral', num_betas = 10, batch_size = 1).to(device)

    with open(hsfm_pkl, 'rb') as f:
        hsfm_output = pickle.load(f)

    # convert to torch tensor from numpy for human params
    human_params = hsfm_output['hsfm_people(smplx_params)']
    for human_name, optim_target_dict in human_params.items():
        for key, value in optim_target_dict.items():
            human_params[human_name][key] = torch.from_numpy(value).to(device)

    world_env = hsfm_output['hsfm_places_cameras']

    show_optimization_results(world_env, body_model_name, human_params, smplx_layer)

if __name__ == '__main__':
    tyro.cli(main)

