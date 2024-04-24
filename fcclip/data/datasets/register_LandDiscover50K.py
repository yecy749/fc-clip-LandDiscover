import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy
import numpy as np
# change unlabled to background
CLASSES_LandDiscover50K = [ 'background','bare land','grass','pavement','road','tree','water',
                            'agriculture land','buildings','forest land','barren land','urban land',
                            'large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                            'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                            'ground-track-field', 'small-vehicle', 'baseball-diamond',
                            'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                            'container-crane', 'airport', 'helipad', 'chimney',
                            'expressway service area','expresswalltoll station','dam',
                            'golf field','overpass','stadium','train station',
                            'vehicle','windmill' ]
LandDiscover50K_COLORS = [np.random.randint(256, size=3).tolist() for k in CLASSES_LandDiscover50K]
MetadataCatalog.get('LandDiscover_50K').set(
    stuff_colors=LandDiscover50K_COLORS[:],
)

def _get_landdiscover50k_meta():
    classes = CLASSES_LandDiscover50K
    assert len(classes) == 40, len(classes)
    ret = {
        "stuff_classes" : classes,
    }
    return ret

def register_ade20k_150(root):
    root = os.path.join(root, "LandDiscover50K")
    meta = _get_landdiscover50k_meta()
    for name,image_dirname, sem_seg_dirname in [
         ("LandDiscover_50K", "TR_Image", "GT_ID"),
     ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)

        name = name
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='png'))
        MetadataCatalog.get(name).set(
            stuff_classes = meta["stuff_classes"][:],
            image_root=image_dir, 
            seg_seg_root=gt_dir, 
            evaluator_type="sem_seg", 
            ignore_label=0, #**meta,)
            gt_ext='png')

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_150(_root)
