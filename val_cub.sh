export CUDA_VISIBLE_DEVICES=0,1,2,3

python ./tools_cam/test_cam.py --config_file configs/CUB/deit_cam_small_patch16_224.ymal --resume save_path TEST.SAVE_BOXED_IMAGE True MODEL.CAM_THR 0.1