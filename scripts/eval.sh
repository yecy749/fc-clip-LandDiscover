export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets/'
export CUDA_VISIBLE_DEVICES=1
python train_net.py \
  --config-file 'configs/coco/panoptic-segmentation/fcclip/LandDiscover50K_fcclip_convnext_large_eval_FLAIR&Potsdam&FloodNet&FAST.yaml' \
  --eval-only MODEL.WEIGHTS output_fromScratch_100K_Batch8/model_final.pth OUTPUT_DIR ./test_inf