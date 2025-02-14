# -*- coding: utf-8 -*-
# @Time    : 2025/02/12
# @Author  : Hongsuk Choi

import math
import gradio
import os
import os.path as osp
import glob
import numpy as np
import copy
import pickle
import tempfile
import shutil
import tyro
import PIL
import cv2

from pathlib import Path
from scipy.spatial.transform import Rotation

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def transform_keypoints(homogeneous_keypoints: np.ndarray, affine_matrix: np.ndarray):
    # Ensure keypoints is a numpy array
    homogeneous_keypoints = np.array(homogeneous_keypoints)

    # Apply the transformation
    transformed_keypoints = np.dot(affine_matrix, homogeneous_keypoints.T).T
    
    # Round to nearest integer for pixel coordinates
    transformed_keypoints = np.round(transformed_keypoints).astype(int)
    
    return transformed_keypoints

def check_affine_matrix(test_img: PIL.Image, original_image: PIL.Image, affine_matrix: np.ndarray):
    assert affine_matrix.shape == (2, 3)

    # get pixels near the center of the image in the new image space
    # Sample 100 pixels near the center of the image
    w, h = test_img.size
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 4  # Use a quarter of the smaller dimension as the radius

    # Generate random offsets within the circular region
    num_samples = 100
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    r = np.random.uniform(0, radius, num_samples)
    
    # Convert polar coordinates to Cartesian
    x_offsets = r * np.cos(theta)
    y_offsets = r * np.sin(theta)
    
    # Add offsets to the center coordinates and ensure they're within image bounds
    sampled_x = np.clip(center_x + x_offsets, 0, w-1).astype(int)
    sampled_y = np.clip(center_y + y_offsets, 0, h-1).astype(int)
    
    # Create homogeneous coordinates
    pixels_near_center = np.column_stack((sampled_x, sampled_y, np.ones(num_samples)))
    
    # draw the pixels on the image and save it
    test_img_pixels = np.asarray(test_img).copy().astype(np.uint8)
    for x, y in zip(sampled_x, sampled_y):
        test_img_pixels = cv2.circle(test_img_pixels, (x, y), 3, (0, 255, 0), -1)
    PIL.Image.fromarray(test_img_pixels).save('test_new_img_pixels.png')

    transformed_keypoints = transform_keypoints(pixels_near_center, affine_matrix)
    # Load the original image
    original_img_array = np.array(original_image)

    # Draw the transformed keypoints on the original image
    for point in transformed_keypoints:
        x, y = point[:2]
        # Ensure the coordinates are within the image bounds
        if 0 <= x < original_image.width and 0 <= y < original_image.height:
            cv2.circle(original_img_array, (int(x), int(y)), int(3*affine_matrix[0,0]), (255, 0, 0), -1)

    # Save the image with drawn keypoints
    PIL.Image.fromarray(original_img_array).save('test_original_img_keypoints.png')

# hard coding to get the affine transform matrix
def preprocess_and_get_transform(file, size=512, square_ok=False):
    img = PIL.Image.open(file)
    original_width, original_height = img.size
    
    # Step 1: Resize
    S = max(img.size)
    if S > size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*size/S)) for x in img.size)
    img_resized = img.resize(new_size, interp)
    
    # Calculate center of the resized image
    cx, cy = img_resized.size[0] // 2, img_resized.size[1] // 2
    
    # Step 2: Crop
    halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
    if not square_ok and new_size[0] == new_size[1]:
        halfh = 3*halfw//4
    
    img_cropped = img_resized.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    
    # Calculate the total transformation
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height
    
    translate_x = (cx - halfw) / scale_x
    translate_y = (cy - halfh) / scale_y
    
    affine_matrix = np.array([
        [1/scale_x, 0, translate_x],
        [0, 1/scale_y, translate_y]
    ])
    
    return img_cropped, affine_matrix

def get_reconstructed_scene(model, device, silent, image_size, filelist, schedule, niter, scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)

    # get affine transform matrix list
    affine_matrix_list = []
    # img_cropped_list = []
    for file in filelist:
        img_cropped, affine_matrix = preprocess_and_get_transform(file)
        affine_matrix_list.append(affine_matrix)
        # img_cropped_list.append(img_cropped)

    # CHECK the first image
    # test_img = img_cropped_list[0]
    # org_img = PIL.Image.open(filelist[0])
    # check_affine_matrix(test_img, org_img, affine_matrix_list[0])
    # import pdb; pdb.set_trace()


    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        print('final loss: ', loss)
    # get optimized values from scene
    # scene = scene_state.sparse_ga
    rgbimg = scene.imgs   # list of N numpy images with shape (H,W,3) , rgb
    intrinsics = to_numpy(scene.get_intrinsics()) # N intrinsics # (N, 3, 3)
    cams2world = to_numpy(scene.get_im_poses()) # (N,4,4)

    # 3D pointcloud from depthmap, poses and intrinsics
    # pts3d: list of N pointclouds, each shape(H, W,3)
    # confs: list of N confidence scores, each shape(H, W)
    # msk: boolean mask of valid points, shape(H, W)
    pts3d = to_numpy(scene.get_pts3d())
    depths = to_numpy(scene.get_depthmaps())
    msk = to_numpy(scene.get_masks())
    confs = to_numpy([c for c in scene.im_conf])

    return rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list, output


def main(model_path: str = './checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', out_pkl_dir: str = './demo_output/input_dust3r', img_dir: str = './demo_data/input_images/bww_stairs_nov17'):
    # parameters
    device = 'cuda'
    silent = False
    image_size = 512
    # there are image diriectories for each view video
    filelist = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if filelist == []:
        filelist = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

    schedule = 'linear'
    niter = 300
    scenegraph_type = 'complete'
    winsize = 1
    refid = 0
    
    # Load your model here
    from dust3r.model import AsymmetricCroCo3DStereo
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list, output = get_reconstructed_scene(model, device, silent, image_size, filelist, schedule, niter, scenegraph_type, winsize, refid)
    
    # Save the results as a pickle file
    results = {}
    for i, f in enumerate(filelist):
        # strip the extension
        img_name = osp.basename(f).split('.')[0]
        results[img_name] = {
            'rgbimg': rgbimg[i],
            'intrinsic': intrinsics[i],
            'cam2world': cams2world[i],
            'pts3d': pts3d[i],
            'depths': depths[i],
            'msk': msk[i],
            'conf': confs[i],
            'affine_matrix': affine_matrix_list[i],
        }

    total_output = {
        "dust3r_network_output": output,
        "dust3r_ga_output": results
    }
    out_pkl_dir = osp.join(out_pkl_dir, osp.basename(img_dir))
    Path(out_pkl_dir).mkdir(parents=True, exist_ok=True)
    model_name = osp.basename(model_path).split('_')[0].lower()
    output_file = osp.join(out_pkl_dir, f'{model_name}_reconstruction_results_{osp.basename(img_dir)}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(total_output, f)

    print(f"Results saved to {output_file}")



if __name__ == '__main__':
    tyro.cli(main)