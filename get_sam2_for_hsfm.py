# -*- coding: utf-8 -*-
# @Time    : 2025/02/12
# @Author  : Hongsuk Choi

import os
import cv2
import time
import tyro
import shutil
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from sam2.gdsam2_utils.video_utils import create_video_from_images
from sam2.gdsam2_utils.common_utils import CommonUtils
from sam2.gdsam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy


def main(text: str="person.", img_dir: str="./tmp_demo_data", output_dir: str="./tmp_demo_output", vis: bool=False, offload_video_to_cpu: bool=False, async_loading_frames: bool=False):
    """Main function for Grounding-SAM-2

    Args:
        text: text queries need to be lowercased + end with a dot
        img_dir: directory of JPEG frames with filenames like `*_<frame_index>.jpg` // No png files supported by SAM2 but I (Hongsuk) just added some code to copy png to jpg
        output_dir: directory to save the annotated frames
        vis: whether to visualize the results
    """
        
    step = 20 # the step to sample frames for Grounding DINO predictor

    
    """
    Step 1: Environment settings and model initialization
    """
    # use bfloat16 for the entire notebook
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # create the output directory
    output_dir = os.path.join(output_dir, os.path.basename(img_dir))        
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "sam2_mask_data")
    json_data_dir = os.path.join(output_dir, "sam2_json_data")
    result_dir = os.path.join(output_dir, "sam2_result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(img_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        new_img_dir = None
    except:
        print("Only JPEG frames are supported by SAM2, whose file names should be like <frame_index>.jpg")
        print("Copying the frames to a new directory and renaming them to <frame_index>.jpg...")
        if img_dir[-1] == '/':
            img_dir = img_dir[:-1]
        new_img_dir = os.path.join(os.path.dirname(img_dir), f"new_{os.path.basename(img_dir)}")
        CommonUtils.creat_dirs(new_img_dir)

        frame_names.sort()
        new_frame_names = []
        for _, fn in tqdm(enumerate(frame_names)):
            # extract frame index from the filename
            # frame_name is like this: frame_00000.jpg
            frame_idx = int(os.path.splitext(fn)[0].split('_')[-1])

            # copy the frame to the new directory
            new_frame_name = f"{frame_idx:05d}.jpg"
            shutil.copy(os.path.join(img_dir, fn), os.path.join(new_img_dir, new_frame_name))
            new_frame_names.append(new_frame_name)
        
        img_dir = new_img_dir
        frame_names = new_frame_names
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=img_dir, offload_video_to_cpu=offload_video_to_cpu, async_loading_frames=async_loading_frames)

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    objects_count = 0

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    for start_frame_idx in range(0, len(frame_names), step):
    # prompt grounding dino to get the box coordinates on specific frame
        print("start_frame_idx", start_frame_idx)
        # continue
        img_path = os.path.join(img_dir, frame_names[start_frame_idx])
        image = Image.open(img_path)
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

        # run Grounding DINO on the image
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )

        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"] # .cpu().numpy()
        # print("results[0]",results[0])
        OBJECTS = results[0]["labels"]
        if input_boxes.shape[0] != 0:
            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            """
            Step 3: Register each object's positive points to video predictor
            """

            # If you are using point prompts, we uniformly sample positive points based on the mask
            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")


            """
            Step 4: Propagate the video predictor to get the segmentation results for each frame
            """
            objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
            print("objects_count", objects_count)
        else:
            print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
            mask_dict = sam2_masks

        
        if len(mask_dict.labels) == 0:
            mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        else: 
            video_predictor.reset_state(inference_state)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )
            
            video_segments = {}  # output the following {step} frames tracking masks
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

            print("video_segments:", len(video_segments))
        """
        Step 5: save the tracking masks and json files
        """
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

            json_data = frame_masks_info.to_dict()
            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(json_data, f)


    """
    Step 6: Draw the results and save the video
    """
    if vis: 
        frame_rate = 30
        CommonUtils.draw_masks_and_box_with_supervision(img_dir, mask_data_dir, json_data_dir, result_dir)
        output_video_path = os.path.join(output_dir, "sam2_output.mp4")
        create_video_from_images(result_dir, output_video_path, frame_rate=frame_rate)

    if new_img_dir:
        print("Removing the new video directory...")
        shutil.rmtree(new_img_dir)

if __name__ == "__main__":
    tyro.cli(main)