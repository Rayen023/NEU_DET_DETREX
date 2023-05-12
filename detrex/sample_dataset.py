import cv2
import random

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer




register_coco_instances("my_dataset_train", {}, "/home/rayen/projects/NET_DET_DETREX/detrex/datasets/surface/annotations/train.json","/home/rayen/projects/NET_DET_DETREX/detrex/datasets/surface/train/" )
register_coco_instances("my_dataset_val", {},"/home/rayen/projects/NET_DET_DETREX/detrex/datasets/surface/annotations/val.json" ,"/home/rayen/projects/NET_DET_DETREX/detrex/datasets/surface/val/" )


defects_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")


for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=defects_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("image",out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()

