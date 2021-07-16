GPU_ID=$1
NET=${2}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python ./tools_cam/train_cam.py --config_file ./configs/CUB/deit_cam_${NET}_patch16_224.ymal --lr 5e-5 MODEL.CAM_THR 0.1