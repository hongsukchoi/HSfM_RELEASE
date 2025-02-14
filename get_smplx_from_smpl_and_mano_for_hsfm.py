# -*- coding: utf-8 -*-
# @Time    : 2025/02/12
# @Author  : Hongsuk Choi

import os
import os.path as osp
import glob
import pickle
import tyro
import torch
import torch.nn as nn
import smplx
import trimesh
import numpy as np

from scipy.spatial.transform import Rotation as R
from pathlib import Path
from tqdm import tqdm

from joint_names import ORIGINAL_SMPLX_JOINT_NAMES, MANO_RIGHT_REORDER_KEYPOINTS, MANO_LEFT_REORDER_KEYPOINTS

# smplx_right_hand_wrist_joint_idx = ORIGINAL_SMPLX_JOINT_NAMES.index('right_wrist')
# smplx_right_hand_joint_idx = [smplx_right_hand_wrist_joint_idx, *range(40,54+1), *range(71,75+1)]
# right_wrist right_index1 right_index2 right_index3 right_middle1 right_middle2 right_middle3 right_pinky1 right_pinky2 right_pinky3 right_ring1 right_ring2 right_ring3 right_thumb1 right_thumb2 right_thumb3 right_thumb right_index right_middle right_ring right_pinky
# the last 5 joints are presumably finger tips
smplx_right_hand_joint_idx_reordered = [ORIGINAL_SMPLX_JOINT_NAMES.index(x) for x in MANO_RIGHT_REORDER_KEYPOINTS]
# right_wrist right_thumb1 right_thumb2 right_thumb3 right_thumb right_index1 right_index2 right_index3 right_index right_middle1 right_middle2 right_middle3 right_middle right_ring1 right_ring2 right_ring3 right_ring right_pinky1 right_pinky2 right_pinky3 right_pinky 


# smplx_left_hand_wrist_joint_idx = ORIGINAL_SMPLX_JOINT_NAMES.index('left_wrist')
# smplx_left_hand_joint_idx = [smplx_left_hand_wrist_joint_idx, *range(25,39+1), *range(66,70+1)]
# left_wrist left_index1 left_index2 left_index3 left_middle1 left_middle2 left_middle3 left_pinky1 left_pinky2 left_pinky3 left_ring1 left_ring2 left_ring3 left_thumb1 left_thumb2 left_thumb3 left_thumb left_index left_middle left_ring left_pinky
# the last 5 joints are presumably finger tips
smplx_left_hand_joint_idx_reordered = [ORIGINAL_SMPLX_JOINT_NAMES.index(x) for x in MANO_LEFT_REORDER_KEYPOINTS]
# left_wrist left_thumb1 left_thumb2 left_thumb3 left_thumb left_index1 left_index2 left_index3 left_index left_middle1 left_middle2 left_middle3 left_middle left_ring1 left_ring2 left_ring3 left_ring left_pinky1 left_pinky2 left_pinky3 left_pinky

class ShapeConverter():
    def __init__(
        self, 
        essentials_folder='essentials',
        inbm_type='smpl',
        outbm_type='smplx',
        in_num_betas=10,
        del_inbm=False,
        def_outbm=False
    ):
        """
        Class for converting betas between body models (e.g. SMPL to SMPL-X)
        Parameters
        ----------
        essentials_folder: str
            path to essentials folder
        inbm_type: str
            type of input body model
        outbm_type: str
            type of output body model
        """
        super().__init__()

        self.inbm_type = inbm_type
        self.outbm_type = outbm_type
        self.essentials_folder = essentials_folder
        self.in_num_betas = in_num_betas

        assert self.inbm_type in ['smil', 'smpl', 'smpla', 'smplh'], 'Only SMPL to SMPL-X conversion is supported'
        assert self.outbm_type in ['smplx', 'smplxa'], 'Only SMPL to SMPL-X conversion is supported'

        self.smpltosmplx = self.load_smpltosmplx()
        self.inbm,  self.inshapedirs = self.load_body_model(model_type=self.inbm_type, num_betas=self.in_num_betas)
        self.outbm, self.outshapedirs = self.load_body_model(model_type=self.outbm_type)

        if del_inbm:
            del self.inbm
            
        if def_outbm:
            del self.outbm

    def load_smpltosmplx(self):
        smpl_to_smplx_path = osp.join(self.essentials_folder, f'body_models/smpl_to_smplx.pkl')
        smpltosmplx = pickle.load(open(smpl_to_smplx_path, 'rb'))
        matrix = torch.tensor(smpltosmplx['matrix']).float()   
        return matrix             

    def load_body_model(self, model_type, num_betas=None):
        if model_type in ['smpl', 'smplx']:
            model_folder = osp.join(self.essentials_folder, 'body_models')
            bm = smplx.create(model_path=model_folder, model_type=model_type)
            shapedirs = bm.shapedirs
        elif model_type == 'smpla':
            model_path = osp.join(self.essentials_folder, 'body_models/smpla/SMPLA_NEUTRAL.pth')
            bm = torch.load(model_path) 
            shapedirs = bm['smpla_shapedirs']
        elif model_type == 'smil':
            model_path = osp.join(self.essentials_folder, 'body_models/smil/smil_packed_info.pth')
            bm = torch.load(model_path) 
            shapedirs = bm['shapedirs']
        elif model_type == 'smplxa':
            model_folder = osp.join(self.essentials_folder, 'body_models')
            kid_template = osp.join(model_folder, 'smil/smplx_kid_template.npy')
            bm = smplx.create(
                model_path=model_folder, model_type='smplx',
                kid_template_path=kid_template, age='kid')
            shapedirs = bm.shapedirs
        elif model_type == 'smplh':
            model_path = osp.join(self.essentials_folder, 'body_models/smplh/neutral/model.npz')
            bm_data = np.load(model_path)
            shapedirs = torch.from_numpy(bm_data['shapedirs']).float()
            # ToDo fix with smplh
            bm = None
        else:
            raise ValueError(f'Unknown model type {model_type}')

        if num_betas is not None:
            shapedirs = shapedirs[:,:,:num_betas]

        return bm, shapedirs

    def forward(self, in_betas):
        """ Convert betas from input to output body model. """
        bs = in_betas.shape[0]

        # get shape blend shapes of input
        assert in_betas.shape[1] == self.inshapedirs.shape[-1], 'Input betas do not match input body model'
        in_shape_displacement = smplx.lbs.blend_shapes(in_betas, self.inshapedirs)

        # find the vertices common between the in- and output model and map them
        in_shape_displacement = torch.einsum('nm,bmv->bnv', self.smpltosmplx, in_shape_displacement)
        in_shape_displacement = in_shape_displacement.view(bs, -1)
        out_shapedirs = self.outshapedirs.reshape(-1, self.outshapedirs.shape[-1])

        # solve for betas in least-squares sense
        lsq_arr = torch.matmul(torch.inverse(torch.matmul(
            out_shapedirs.t(), out_shapedirs)), out_shapedirs.t())

        out_betas = torch.einsum('ij,bj->bi', [lsq_arr, in_shape_displacement])

        return out_betas
    
def rotation_matrix_to_angle_axis(x):
    # x: (N, 3, 3)
    r = R.from_matrix(x)
    return r.as_rotvec() # (N, 3)

def conv_mano_to_smplx_hand_pose(x, is_right):
    # x: (15, 3)
    # if is_right == 0.0: # left hand
    #     x = x.reshape(-1)
    #     x[1::3] *= -1
    #     x[2::3] *= -1

    #     x = x.reshape(-1, 3) # (15, 3)

    return x

def conv_smpl_to_smplx_body_pose(x):
    # x: (23, 3)
    # exclude the last 2 joints; 'left_hand_root', 'right_hand_root'
    return x[:-2]

def conv_body_shape(x, shape_converter):
    # x: (10)
    x = torch.from_numpy(x).float().unsqueeze(0)
    with torch.no_grad():
        x = shape_converter.forward(x.cpu()).float().numpy()
    return x[0]

def main(
        mano_dir: str='./demo_data/input_3d_meshes/arthur_tyler_pass_by_nov20/cam01',
        smpl_dir: str='./demo_data/input_3d_meshes/arthur_tyler_pass_by_nov20/cam01', 
        output_dir: str='./demo_data/input_3d_meshes/arthur_tyler_pass_by_nov20/cam01',
        vis: bool=False,
    ):

    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, os.path.basename(mano_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mano_files = sorted(glob.glob(os.path.join(mano_dir, 'mano_*.pkl')))
    smpl_files = sorted(glob.glob(os.path.join(smpl_dir, 'smpl_*.pkl')))

    assert len(mano_files) == len(smpl_files), f"Number of mano files ({len(mano_files)}) and smpl files ({len(smpl_files)}) do not match"

    smpl_to_smplx_shape_converter = ShapeConverter(essentials_folder='./',
            inbm_type='smpl', outbm_type='smplx', def_outbm=True, del_inbm=True)

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    smplx_layer = smplx.create(model_path = './body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = 1).float().to(device)

    if vis:
        vis_output_dir = os.path.join(output_dir, 'vis')
        Path(vis_output_dir).mkdir(parents=True, exist_ok=True)
        
    for mano_file, smpl_file in tqdm(zip(mano_files, smpl_files)):
        frame_idx = os.path.basename(mano_file).split('.')[0].split('_')[1]
        
        assert frame_idx == os.path.basename(smpl_file).split('.')[0].split('_params_')[1], f"Frame index mismatch between {mano_file} and {smpl_file}"

        with open(mano_file, 'rb') as f:
            mano_data = pickle.load(f)
        with open(smpl_file, 'rb') as f:
            smpl_data = pickle.load(f)

        # Dummy template
        smplx_params = {
            'body_pose': [], 
            'betas': [], 
            'global_orient': [], 
            'right_hand_pose': [], 
            'left_hand_pose': [], 
        }

        result_dict = {}
        # Combine smpl and mano   
        for person_id in mano_data.keys():
            right_hand_mano_data = mano_data[person_id]['right_hand'] # could be None
            left_hand_mano_data = mano_data[person_id]['left_hand'] # could be None
            smpl_params = smpl_data[person_id]['smpl_params']

            # smpl_params: 'global_orient': (1, 3, 3), 'body_pose': (23, 3, 3), 'betas': (10)
            # hand_pose: (15, 3)

            if right_hand_mano_data is not None:
                converted_right_hand_pose = conv_mano_to_smplx_hand_pose(right_hand_mano_data['hand_pose'], 1.0)
            else:
                converted_right_hand_pose = None

            if left_hand_mano_data is not None:
                converted_left_hand_pose = conv_mano_to_smplx_hand_pose(left_hand_mano_data['hand_pose'], 0.0)
            else:
                converted_left_hand_pose = None

            converted_betas = conv_body_shape(smpl_params['betas'], smpl_to_smplx_shape_converter)
            converted_body_pose = conv_smpl_to_smplx_body_pose(smpl_params['body_pose']) # (21, 3, 3)
            converted_global_orient = smpl_params['global_orient']

            # convert 3by3 rotation matrix to axis angle
            converted_global_orient_axis_angle = rotation_matrix_to_angle_axis(converted_global_orient) # (1, 3)
            converted_body_pose_axis_angle = rotation_matrix_to_angle_axis(converted_body_pose) # (21, 3)
            
            smplx_params = dict(
                global_orient = converted_global_orient_axis_angle, # (1, 3)
                body_pose = converted_body_pose_axis_angle, # (21, 3)
                betas = converted_betas, # (10)
                left_hand_pose = converted_left_hand_pose, # (15, 3)
                right_hand_pose = converted_right_hand_pose, # (15, 3)
            )

            optimizing_params = {}
            global_orient = torch.from_numpy(smplx_params['global_orient']).to(device).float().reshape(1, -1)
            body_pose = torch.from_numpy(smplx_params['body_pose']).to(device).float().reshape(1, -1) # (1, 21*3) 21 joints except the first pevlis joint
            betas = torch.from_numpy(smplx_params['betas']).to(device).float().reshape(1, -1)
            if smplx_params['left_hand_pose'] is not None:
                left_hand_pose = torch.from_numpy(smplx_params['left_hand_pose']).to(device).float().reshape(1, -1)
                # left wrist pose is the 19th joint in the body pose
                optimizing_params['left_wrist_pose'] = nn.Parameter(body_pose[0:1, 19*3:(19+1)*3])
            else:
                left_hand_pose = None

            if smplx_params['right_hand_pose'] is not None:
                right_hand_pose = torch.from_numpy(smplx_params['right_hand_pose']).to(device).float().reshape(1, -1)
                # right wrist pose is the 20th joint in the body pose
                optimizing_params['right_wrist_pose'] = nn.Parameter(body_pose[0:1, 20*3:(20+1)*3])
            else:
                right_hand_pose = None

            # Save original mesh
            if vis:
                smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=torch.zeros(1,3).to(device))

                vertices = smplx_output['vertices'].detach().cpu().numpy()[0]
                mesh_base_color=(1.0, 1.0, 0.9)
                vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])

                mesh = trimesh.Trimesh(vertices.copy(), smplx_layer.faces.copy(), vertex_colors=vertex_colors)

                saving_path = os.path.join(vis_output_dir, f'{frame_idx}_smplx_person_{person_id:02d}_original.obj')
                mesh.export(saving_path)
                print(f"Saved to {saving_path}")

            if len(optimizing_params) > 0:
                # print(f"Optimizing {len(optimizing_params)} parameters; ", optimizing_params)
                # Get optimizer
                # Create the LBFGS optimizer
                optimizer = torch.optim.LBFGS(
                    list(optimizing_params.values()),
                    lr=1.0,               # learning rate
                    max_iter=10,         # maximum number of iterations per .step() call
                    line_search_fn='strong_wolfe'  # type of line search
                )

                if left_hand_pose is not None:
                    mano_left_hand_joints = torch.from_numpy(mano_data[person_id]['left_hand']['pred_keypoints_3d']).to(device).float().detach() # (21, 3)
                    mano_left_hand_joints = mano_left_hand_joints - mano_left_hand_joints[0:1, :]
                    body_pose = torch.cat([body_pose[:, :19*3], optimizing_params['left_wrist_pose'], body_pose[:, (19+1)*3:]], dim=1)
                if right_hand_pose is not None:
                    mano_right_hand_joints = torch.from_numpy(mano_data[person_id]['right_hand']['pred_keypoints_3d']).to(device).float().detach() # (21, 3)
                    mano_right_hand_joints = mano_right_hand_joints - mano_right_hand_joints[0:1, :]
                    body_pose = torch.cat([body_pose[:, :20*3], optimizing_params['right_wrist_pose'], body_pose[:, (20+1)*3:]], dim=1)

                def create_closure():
                    def closure():
                        optimizer.zero_grad()
                        body_pose = torch.from_numpy(smplx_params['body_pose']).to(device).float().reshape(1, -1) # (1, 21*3) 21 joints except the first pevlis joint

                        if left_hand_pose is not None:
                            body_pose = torch.cat([body_pose[:, :19*3], optimizing_params['left_wrist_pose'], body_pose[:, (19+1)*3:]], dim=1)
                        if right_hand_pose is not None:
                            body_pose = torch.cat([body_pose[:, :20*3], optimizing_params['right_wrist_pose'], body_pose[:, (20+1)*3:]], dim=1)

                        smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
                        smplx_joints = smplx_output['joints'][0] # (144, 3)
                        smplx_right_hand_joints = smplx_joints[smplx_right_hand_joint_idx_reordered, :] # (21, 3)
                        smplx_left_hand_joints = smplx_joints[smplx_left_hand_joint_idx_reordered, :] # (21, 3)

                        loss = 0.0
                        if right_hand_pose is not None:
                            # root (wrist joint) align the joints
                            smplx_right_hand_joints = smplx_right_hand_joints - smplx_right_hand_joints[0:1, :]
                            loss += (smplx_right_hand_joints - mano_right_hand_joints).pow(2).sum()
                        if left_hand_pose is not None:
                            smplx_left_hand_joints = smplx_left_hand_joints - smplx_left_hand_joints[0:1, :]
                            loss += (smplx_left_hand_joints - mano_left_hand_joints).pow(2).sum()

                        loss.backward()                
                        return loss
                    
                    return closure
                
                # closure_fn = create_closure(optimizer, right_hand_pose, left_hand_pose, betas, global_orient, optimizing_params)
                closure_fn = create_closure()
                num_iter = 10
                loss_prev = 0.0
                loss_thresh = 1e-3
                for _ in range(num_iter):
                    loss = optimizer.step(closure_fn)
                    # print(f"Loss: {loss.item():.4f}")
                    # if loss is not reducing, break
                    if loss.item() - loss_prev < loss_thresh:
                        break
                    loss_prev = loss.item()

            # print("Optimized parameters: ", optimizing_params)
            if left_hand_pose is not None:
                body_pose = torch.cat([body_pose[:, :19*3], optimizing_params['left_wrist_pose'], body_pose[:, (19+1)*3:]], dim=1)
            if right_hand_pose is not None:
                body_pose = torch.cat([body_pose[:, :20*3], optimizing_params['right_wrist_pose'], body_pose[:, (20+1)*3:]], dim=1)

            smplx_params['body_pose'] = body_pose.detach().cpu().numpy().reshape(21, 3)
            result_dict[person_id] = smplx_params


            if vis:
                # Save mesh
                print("Saving mesh")
                smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=torch.zeros(1,3).to(device))

                vertices = smplx_output['vertices'].detach().cpu().numpy()[0]
                mesh_base_color=(1.0, 1.0, 0.9)
                vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])

                mesh = trimesh.Trimesh(vertices.copy(), smplx_layer.faces.copy(), vertex_colors=vertex_colors)

                # saving_path = os.path.join(vis_output_dir, f'{frame_idx}_smplx_person_{person_id:02d}_rot_{rot_idx:02d}.obj')
                saving_path = os.path.join(vis_output_dir, f'{frame_idx}_smplx_person_{person_id:02d}.obj')
                mesh.export(saving_path)

                print(f"Saved to {saving_path}")

        with open(os.path.join(output_dir, f'smplx_params_{frame_idx}.pkl'), 'wb') as f:
            pickle.dump(result_dict, f)
        print(f"Saved smplx_params_{frame_idx}.pkl to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)