conda create --name fcclip python=3.8 -y
conda activate fcclip
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
git clone https://github.com/bytedance/fcclip.git
cd fcclip
pip install -r requirements.txt
cd fcclip/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..