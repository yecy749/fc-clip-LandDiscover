export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets/'
# export CUDA_VISIBLE_DEVICES=1
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_landdiscover50k.yaml \
  --num-gpus 2 SOLVER.IMS_PER_BATCH 10