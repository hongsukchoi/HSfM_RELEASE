# Egohumans 2D annotation joint names
# https://github.com/open-mmlab/mmhuman3d/blob/main/mmhuman3d/core/conventions/keypoints_mapping/coco_wholebody.py
# https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png
COCO_WHOLEBODY_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'right_contour_1',  # original name: face_contour_1
    'right_contour_2',  # original name: face_contour_2
    'right_contour_3',  # original name: face_contour_3
    'right_contour_4',  # original name: face_contour_4
    'right_contour_5',  # original name: face_contour_5
    'right_contour_6',  # original name: face_contour_6
    'right_contour_7',  # original name: face_contour_7
    'right_contour_8',  # original name: face_contour_8
    'contour_middle',  # original name: face_contour_9
    'left_contour_8',  # original name: face_contour_10
    'left_contour_7',  # original name: face_contour_11
    'left_contour_6',  # original name: face_contour_12
    'left_contour_5',  # original name: face_contour_13
    'left_contour_4',  # original name: face_contour_14
    'left_contour_3',  # original name: face_contour_15
    'left_contour_2',  # original name: face_contour_16
    'left_contour_1',  # original name: face_contour_17
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',# 'nosebridge_1',
    'nose2',# 'nosebridge_2',
    'nose3',# 'nosebridge_3',
    'nose4',# 'nosebridge_4',
    'right_nose_2',  # original name: nose_1
    'right_nose_1',  # original name: nose_2
    'nose_middle',  # original name: nose_3
    'left_nose_1',  # original name: nose_4
    'left_nose_2',  # original name: nose_5
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',  # original name: mouth_1
    'right_mouth_2',  # original name: mouth_2
    'right_mouth_3',  # original name: mouth_3
    'mouth_top',  # original name: mouth_4
    'left_mouth_3',  # original name: mouth_5
    'left_mouth_2',  # original name: mouth_6
    'left_mouth_1',  # original name: mouth_7
    'left_mouth_5',  # original name: mouth_8
    'left_mouth_4',  # original name: mouth_9
    'mouth_bottom',  # original name: mouth_10
    'right_mouth_4',  # original name: mouth_11
    'right_mouth_5',  # original name: mouth_12
    'right_lip_1',  # original name: lip_1
    'right_lip_2',  # original name: lip_2
    'lip_top',  # original name: lip_3
    'left_lip_2',  # original name: lip_4
    'left_lip_1',  # original name: lip_5
    'left_lip_3',  # original name: lip_6
    'lip_bottom',  # original name: lip_7
    'right_lip_3',  # original name: lip_8
    'left_hand_root',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'left_thumb',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_index',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_middle',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_ring',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_pinky',
    'right_hand_root',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'right_thumb',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_index',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_middle',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_ring',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_pinky',
]

# https://github.com/open-mmlab/mmhuman3d/blob/main/mmhuman3d/core/conventions/keypoints_mapping/mano.py
# Original order from MANO J_regressor
MANO_RIGHT_KEYPOINTS = [
    'right_wrist', 'right_index1', 'right_index2', 'right_index3',
    'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1',
    'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2',
    'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3',
    'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
]

MANO_LEFT_KEYPOINTS = [
    x.replace('right_', 'left_') for x in MANO_RIGHT_KEYPOINTS
]

# Re-arranged order is compatible with the output of manolayer
# from official [manopth](https://github.com/hassony2/manopth)
MANO_REORDER_MAP = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]

MANO_RIGHT_REORDER_KEYPOINTS = [
    MANO_RIGHT_KEYPOINTS[i] for i in MANO_REORDER_MAP
]
MANO_LEFT_REORDER_KEYPOINTS = [
    MANO_LEFT_KEYPOINTS[i] for i in MANO_REORDER_MAP
]


# MultiHMR SMPLX joint names
SMPLX_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip', 
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',
    'left_mouth_4',
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3'
] # ORIGINAL_SMPLX_JOINT_NAMES[:127]

SMPL_KEYPOINTS = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand_root',
    'right_hand_root',
]

# the full keypoints produced by the default SMPL J_regressor
SMPL_45_KEYPOINTS = SMPL_KEYPOINTS + [
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]

# Original SMPLX joint names - https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
# face joint order: https://github.com/Rubikplayer/flame-fitting
ORIGINAL_SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

# 135 joints that George and Lea are using. 
# neck and mid_hip here are not in the COCO_WHOLEBODY_KEYPOINTS
VITPOSEPLUS_KEYPOINTS = [ 
    'nose',
    'neck', 
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow', 
    'left_wrist',
    'mid_hip',
    'right_hip',
    'right_knee',
    'right_ankle', 
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_wrist_openpose',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'left_thumb',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_index',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_middle',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_ring',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_pinky',
    'right_wrist_openpose',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'right_thumb',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_index',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_middle',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_ring',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',
    'left_mouth_4',
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1'
]
# ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist', 'mid_hip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_wrist', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb', 'left_index1', 'left_index2', 'left_index3', 'left_index', 'left_middle1', 'left_middle2', 'left_middle3', 'left_middle', 'left_ring1', 'left_ring2', 'left_ring3', 'left_ring', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky', 'right_wrist', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb', 'right_index1', 'right_index2', 'right_index3', 'right_index', 'right_middle1', 'right_middle2', 'right_middle3', 'right_middle', 'right_ring1', 'right_ring2', 'right_ring3', 'right_ring', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky', 'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'left_mouth_5', 'left_mouth_4', 'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3', 'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1']




# Define skeleton edges using indices of main body joints
COCO_MAIN_BODY_SKELETON = [
    # Torso
    [5, 6],   # left_shoulder to right_shoulder
    [5, 11],  # left_shoulder to left_hip
    [6, 12],  # right_shoulder to right_hip
    [11, 12], # left_hip to right_hip
    
    # Left arm
    [5, 7],   # left_shoulder to left_elbow
    [7, 9],   # left_elbow to left_wrist
    
    # Right arm
    [6, 8],   # right_shoulder to right_elbow
    [8, 10],  # right_elbow to right_wrist
    
    # Left leg
    [11, 13], # left_hip to left_knee
    [13, 15], # left_knee to left_ankle
    [15, 19], # left_ankle to left_heel
    
    # Right leg
    [12, 14], # right_hip to right_knee
    [14, 16], # right_knee to right_ankle
    [16, 22], # right_ankle to right_heel

    # Head
    [0, 1], # nose to left_eye
    [0, 2], # nose to right_eye
    [1, 3], # left_eye to left_ear
    [2, 4], # right_eye to right_ear
]

