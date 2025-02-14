#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 --img-dir <image_directory> --out-dir <output_directory> [--vis] [--person-ids \"1 2 3\"]"
    exit 1
fi

# Default values
VIS=""
PERSON_IDS="1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --img-dir)
            IMG_DIR="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --vis)
            VIS="--vis"
            shift
            ;;
        --person-ids)
            PERSON_IDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Remove trailing slashes from directories
IMG_DIR=${IMG_DIR%/}
OUT_DIR=${OUT_DIR%/}

# Extract the base name of the image directory
IMG_DIR_NAME=$(basename "$IMG_DIR")

echo "Running HSfM pipeline..."
echo "Image directory: $IMG_DIR"
echo "Output directory: $OUT_DIR"
echo "Visualization: ${VIS:+enabled}"
echo "Person IDs: $PERSON_IDS"

# Run the pipeline
python get_world_env_dust3r_for_hsfm.py \
    --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    --img-dir "$IMG_DIR" \
    --out-pkl-dir "$OUT_DIR"

# python vis_viser_dust3r.py \
#     --world-env-pkl "$OUT_DIR/$IMG_DIR_NAME/dust3r_reconstruction_results_${IMG_DIR_NAME}.pkl" \
#     --world_scale_factor 5.0 \
#     --conf-thr 1.5

python get_sam2_for_hsfm.py \
    --img-dir "$IMG_DIR" \
    --output-dir "$OUT_DIR" \
    $VIS

python get_pose2d_vitpose_for_hsfm.py \
    --img-dir "$IMG_DIR" \
    --bbox-dir "$OUT_DIR/$IMG_DIR_NAME/sam2_json_data" \
    --output-dir "$OUT_DIR" \
    $VIS

python get_smpl_hmr2_for_hsfm.py \
    --img-dir "$IMG_DIR" \
    --bbox-dir "$OUT_DIR/$IMG_DIR_NAME/sam2_json_data" \
    --output-dir "$OUT_DIR" \
    --person-ids $PERSON_IDS \
    $VIS

python get_mano_wilor_for_hsfm.py \
    --img-dir "$IMG_DIR" \
    --pose2d-dir "$OUT_DIR/$IMG_DIR_NAME" \
    --output-dir "$OUT_DIR" \
    --person-ids $PERSON_IDS \
    $VIS

python get_smplx_from_smpl_and_mano_for_hsfm.py \
    --smpl-dir "$OUT_DIR/$IMG_DIR_NAME" \
    --mano-dir "$OUT_DIR/$IMG_DIR_NAME" \
    --output-dir "$OUT_DIR" \
    $VIS

python align_world_env_and_smpl_hsfm_optim.py \
    --world-env-path "$OUT_DIR/$IMG_DIR_NAME/dust3r_reconstruction_results_${IMG_DIR_NAME}.pkl" \
    --person-ids $PERSON_IDS \
    --bbox-dir "$OUT_DIR/$IMG_DIR_NAME/sam2_json_data" \
    --pose2d-dir "$OUT_DIR/$IMG_DIR_NAME" \
    --smplx-dir "$OUT_DIR/$IMG_DIR_NAME" \
    --body-model-name smplx \
    --out-dir "$OUT_DIR/$IMG_DIR_NAME" \

echo "HSfM pipeline completed!" 