# Human and Structure from Motion (HSfM) 
[[project page]](https://muelea.github.io/hsfm/) [[arxiv]](https://arxiv.org/abs/2412.17806)

<div style="background-color:#222; padding:10px; border-radius:5px; color:white;">
Lea Muller*, Hongsuk Choi*, Anthony Zhang, Brent Yi, Jitendra Malik, Angjoo Kanazawa.
<br>
<b> Reconstructing People, Places, and Cameras.</b> arXiv, 2024.
<br>
<i>UC Berkeley</i>
<br>
<i>*Equal contribution</i>
</div>

## Run HSfM at one script

```bash
# With specific person IDs; ids are predicted by Grounding-SAM-2; default is a person with id 1
./run_hsfm.sh --img-dir ./demo_data/people_jumping --out-dir ./demo_output --person-ids "1 2 3" --vis

# Visualize the HSfM result
python vis_viser_hsfm.py --hsfm-pkl ./demo_output/people_jumping/hsfm_output_smplx.pkl
```

## Updates

- Feb 15, 2025: Released the demo code to run HSfM on custom data.

## Overview
**TL;DR;** HSfM jointly reconstructs people, places, and cameras.  
> Intuition: Humans aid camera and scene structure estimation, while the camera and scene structure, in turn, enhance human reconstruction.

![HSfM](./assets/teaser.gif)
  


## HSfM commands

Human and Structure from Motion (HSfM) pipeline involves multiple steps.  
1. We assume correct reidentification of people (matching id of people across different views) is given.
2. Regardlessly of the method of reidentification (ReID), you can use Grounding-SAM-2 to get ReID and bounding boxes of people. Masks are also provided by Grounding-SAM-2, but they are not used in the pipeline. If ReID output from Grounding-SAM-2 is not good, you can use other ReID methods to get ReID and bounding boxes of people, or manually fix the ids. [How to fix the ids](#how-to-fix-the-ids)
3. We use ViTPose to get 2D poses of people.
4. We use HMR2.0 to get 3D human mesh. Make sure to feed correct person ids obtained from Grounding-SAM-2.
5. We use WiLor to get 3D hand mesh. Make sure to feed correct person ids obtained from Grounding-SAM-2.
6. We combine a SMPL body mesh from HMR2.0 and a MANO hand mesh from WiLor to get a SMPL-X 3D whole body (hand + body) mesh.
7. We jointly optimize people, places, and cameras. Make sure to feed correct person ids obtained from Grounding-SAM-2.

```bash
# Run Grounding-SAM-2 to get reid and bbox of people; set vis to True to visualize the result; Check the reidentification result before running the pipeline
python get_sam2_for_hsfm.py --img-dir ./demo_data/people_jumping --output-dir ./demo_output --vis

# Run dust3r to get world environment
python get_world_env_dust3r_for_hsfm.py --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth  --img-dir ./demo_data/people_jumping --out-pkl-dir ./demo_output

# Visualize the world environment from dust3r
python vis_viser_dust3r.py --world-env-pkl ./demo_output/people_jumping/dust3r_reconstruction_results_people_jumping.pkl --world_scale_factor 5.0 --conf-thr 1.5  # world-scale-factor is just for visualization convenience

# Run ViTPose to get 2D poses of people
python get_pose2d_vitpose_for_hsfm.py --img-dir ./demo_data/people_jumping --bbox-dir ./demo_output/people_jumping/sam2_json_data --output-dir ./demo_output --vis # when running this, it's fine to see model weights warning

# Run HMR2.0 to get 3D human mesh
python get_smpl_hmr2_for_hsfm.py --img-dir ./demo_data/people_jumping --bbox-dir ./demo_output/people_jumping/sam2_json_data --output-dir  ./demo_output/ --person-ids 1 2 3 --vis # choose people you want to use for optimization

# Run WiLor to get 3D hand mesh
python get_mano_wilor_for_hsfm.py --img-dir ./demo_data/people_jumping --pose2d-dir ./demo_output/people_jumping --output-dir ./demo_output --person-ids 1 2 3 --vis

# Get SMPL-X from SMPL and MANO
python get_smplx_from_smpl_and_mano_for_hsfm.py --smpl-dir ./demo_output/people_jumping --mano-dir ./demo_output/people_jumping --output-dir ./demo_output/ --vis

# Align world environment and SMPL-X and optimize
python align_world_env_and_smpl_hsfm_optim.py --world-env-path ./demo_output/people_jumping/dust3r_reconstruction_results_people_jumping.pkl  --person-ids 1 2 3 --bbox-dir ./demo_output/people_jumping/sam2_json_data --pose2d-dir ./demo_output/people_jumping  --smplx-dir ./demo_output/people_jumping --body-model-name smplx --out-dir ./demo_output/people_jumping  --vis
# if you set vis, you will first see HSfM (initialization) in viser;
# after checking the result, cancel it by ctrl+c, then the process will continue to HSfM optimization

# Visualize the HSfM result
python vis_viser_hsfm.py --hsfm-pkl ./demo_output/people_jumping/hsfm_output_smplx.pkl
```

Some tips and information:
- Initialization is important! If something goes wrong, check the visualization of each step in viser by setting `vis` on `align_world_env_and_smpl_hsfm_optim.py`
- you can skip the mano and smplx process, just pass 'smpl' to body-model-name argument in align_world_env_and_smpl_hsfm_optim.py
- setting vis on for each process takes much more time to process
- most of the time is spent on creating the model and loading the model weights
- becareful to not include '/' at the end of the directory argument, it will make issues

You might want to tune the following parameters:
```python
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
```

## How to fix ids

- If the ids predicted by Grounding-SAM-2 are not good, you can use other ReID methods to get ReID and bounding boxes of people, or manually fix the ids.
- But, you should the follow the output format of Grounding-SAM-2. The `mask_name`, `mask_height`, `mask_width`, `promote_type` are unncessary for the pipeline, you should the format of the `labels` key/value pairs and the corresponding `instance_id`.  
Example:  
```json
{"mask_name": "mask_00001.npy", "mask_height": 1280, "mask_width": 720, "promote_type": "mask", "labels": {"1": {"instance_id": 1, "class_name": "person", "x1": 455, "y1": 261, "x2": 681, "y2": 811, "logit": 0.0}, "2": {"instance_id": 2, "class_name": "person", "x1": 15, "y1": 78, "x2": 236, "y2": 666, "logit": 0.0}, "3": {"instance_id": 3, "class_name": "person", "x1": 309, "y1": 148, "x2": 492, "y2": 651, "logit": 0.0}}}
```


## Installation

```bash
conda create -n hsfm python=3.10 # python<=3.10 is recommended because of ViTPose installation issues

pip install git+https://github.com/hongsukchoi/dust3r.git
pip install git+https://github.com/hongsukchoi/croco.git


# Grounding-SAM-2 to get bonuding box and mask
git clone https://github.com/hongsukchoi/Grounded-SAM-2.git
cd Grounded-SAM-2
export CUDA_HOME=/usr/local/cuda-12.1 # or your cuda version!
pip install -e . # Install Segment Anything 2
pip install --no-build-isolation -e grounding_dino # Install Grounding DINO

# ViTPose to get 2d poses
pip install -U openmim
mim install mmcv==1.3.9
cd ..
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .

# HMR2.0 to get 3D human mesh
pip install git+https://github.com/hongsukchoi/4D-Humans.git

# WiLor to get 3D hand mesh
pip install git+https://github.com/warmshao/WiLoR-mini

pip install -r requirements.txt 
```

<details>
<summary>If you have problems regarding g+-11 when installing Grounding DINO, try this:</summary>

```bash
# Install g++-11
sudo apt update
sudo apt install g++-11
# Set environment variables to use g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install --no-build-isolation -e grounding_dino # Install Grounding DINO
```
</details>


## Directory structure 
```
assets/
body_models/
    smpl/
        basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl # Download SMPL from here: https://smplify.is.tue.mpg.de
        SMPL_NEUTRAL.pkl # Download SMPL from here: https://smplify.is.tue.mpg.de   
    smplx/
        SMPLX_NEUTRAL.npz # Download SMPL-X from here: https://smpl-x.is.tue.mpg.de
    mano/
        MANO_RIGHT.pkl # Download MANO from here: https://mano.is.tue.mpg.de
    smpl_to_smplx.pkl # Download smpl_to_smplx.pkl from my google drive: https://drive.google.com/file/d/1Avg_lNdWmVAF3sljxpAj7rnTK8RfGKZi/view?usp=sharing
checkpoints/
    DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth # Download dust3r checkpoint from here: https://github.com/hongsukchoi/dust3r/tree/main?tab=readme-ov-file#checkpoints
    vitpose_huge_wholebody.pth # Download ViTPose checkpoint from here: https://github.com/ViTAE-Transformer/ViTPose/tree/main?tab=readme-ov-file#checkpoints
    sam2.1_hiera_large.pt # Download Grounding-SAM-2 checkpoint from here: https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file#installation
configs/
    vitpose/
        ...
demo_data/
    people_jumping/
        000001.jpg # first view
        000002.jpg # second view
        ...
demo_output/
    people_jumping/
        sam2_json_data/
            mask_00001.json # reid and bbox information of the first view
            mask_00002.json # reid and bbox information of the second view
        sam2_mask_data/
            mask_00001.npy # human mask of the first view
            mask_00002.npy # human mask of the second view
            ...
        sam2_result/
            000001.jpg # first view gsam2 result
            000002.jpg # second view gsam2 result
            ...
        sam2_output.mp4
        dust3r_reconstruction_results_people_jumping.pkl # pointmaps and cameras in the world coordinate system from dust3r
        mano_00001.pkl # 3D hand mesh of the first view
        mano_00002.pkl # 3D hand mesh of the second view
        ...
        pose_00001.json # 2D pose of the first view
        pose_00002.json # 2D pose of the second view
        ...
        smpl_params_00001.pkl # SMPL parameters of the first view
        smpl_params_00002.pkl # SMPL parameters of the second view
        ...
        smplx_params_00001.pkl # SMPL-X parameters of the first view
        smplx_params_00002.pkl # SMPL-X parameters of the second view
        ...
        hsfm_output_smplx.pkl # HSfM result
```

- Download body models following the instructions in the links above.
- Download checkpoints from here: https://drive.google.com/drive/folders/1Si2tRx9AQrqno1Q7V2LemHdM10B1Hy1d?usp=sharing 

