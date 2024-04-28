export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets/'
export CUDA_VISIBLE_DEVICES=1,0
python train_net.py \
  --config-file "configs/coco/panoptic-segmentation/fcclip/LandDiscover50K_fcclip_convnext_large_eval_FLAIR&Potsdam&FloodNet&FAST.yaml" \
  --num-gpus 2 --resume \
  OUTPUT_DIR /media/zpp2/PHDD/FC-CLIP-output/output_FT_100K_Batch8 MODEL.WEIGHTS fcclip_cocopan.pth SOLVER.IMS_PER_BATCH 8 SOLVER.MAX_ITER 100000 DATALOADER.NUM_WORKERS 16 \
  