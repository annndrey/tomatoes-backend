#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch, torchvision

# detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,Visualizer_with_labeled_colors
from detectron2.data import MetadataCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode


########
from detectron2.data.datasets import register_coco_instances

def predict(path2read,
            path2save):
    
    register_coco_instances("leaves_train_cocostyle",{},"/home/imolodtsov/keep_copies/datasets/coco/annotations/leaves_train.json","/home/imolodtsov/keep_copies/datasets/coco/leaves_train/")
    cfg = get_cfg()
    cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("leaves_train_cocostyle",
                          #"GenFromSingleLeafs_cocostyle",
                          #"GenFromSingleLeafs_leafsnap_cocostyle",
                          #"SingleLeaves_web_cocostyle",
                          #"SingleLeafs_cocostyle",
                          #"Jun25_Backgrounds_cocostyle",
                         )
    cfg.DATASETS.TEST = ("leaves_train_cocostyle",)   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2 # Number of data loading threads
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2 #Images per Batch; E.g. if we have 16 GPUs and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
    cfg.SOLVER.BASE_LR = 0.00025 #LR - learning rate
    cfg.SOLVER.MAX_ITER = 50    # number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # number of proposals to sample when training
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (leaf)

    cfg.OUTPUT_DIR="/home/imolodtsov/detectron2/trained_models/mask_rcnn_X_101_32x8d_FPN_3x"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("leaves_test_cocostyle", )
    cfg.MODEL.DEVICE="cpu"
    predictor = DefaultPredictor(cfg)

    MetadataCatalog.get('/large_disc/pics_for_fermata/coco/leaves_test/').set(thing_classes=["leaf"])
    leaf_metadata = MetadataCatalog.get('/large_disc/pics_for_fermata/coco/leaves_test/')

    

    im = cv2.imread(path2read)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=leaf_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                   
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(path2save,v.get_image()[:, :, ::-1])

if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
    #predict("/large_disc/pics_for_fermata/potato_from_VNIIF/alternarioz_a/20190729_122240.jpg","/home/imolodtsov/data/temp.jpg")
