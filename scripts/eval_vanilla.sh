export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets/'
export CUDA_VISIBLE_DEVICES=1
python train_net.py \
  --config-file 'configs/coco/panoptic-segmentation/fcclip/LandDiscover50K_fcclip_convnext_large_eval_FLAIR&Potsdam&FloodNet&FAST.yaml' \
  --eval-only MODEL.WEIGHTS /home/zpp2/ycy/fc-clip/fcclip_cocopan.pth OUTPUT_DIR ./vanilla_inf SOLVER.IMS_PER_BATCH 8