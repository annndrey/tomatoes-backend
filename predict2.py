#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch, torchvision
import sys
# detectron2 logger
import detectron2
#from detectron2.utils.logger import setup_logger
#setup_logger()

# import some common libraries
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode,_create_text_labels,GenericMask
from detectron2.data import MetadataCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode


########
from detectron2.data.datasets import register_coco_instances

###################################################################################################################
###################################################################################################################

#temp return randint
def status_predict4image(cv2_im):
    return(np.random.randint(3))

###################################################################################################################
###################################################################################################################
class Visualizer_with_labeled_colors(Visualizer):
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE,clabels=None):
        super().__init__(img_rgb, metadata, scale=scale, instance_mode=instance_mode)
        self.clabels=clabels
    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks.numpy()
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = self.clabels
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            assert predictions.has("pred_masks"), "ColorMode.IMAGE_BW requires segmentations"
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
###################################################################################################################
###################################################################################################################
def create_predict_instance():
    register_coco_instances("leaves_train_cocostyle",{},"/home/imolodtsov/keep_copies/datasets/coco/annotations/leaves_train.json","/home/imolodtsov/keep_copies/datasets/coco/leaves_train/")
    cfg = get_cfg()
    cfg.merge_from_file("/home/imolodtsov/temp_de/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("leaves_train_cocostyle",)
    cfg.DATASETS.TEST = ("leaves_train_cocostyle",)
    
    cfg.DATALOADER.NUM_WORKERS = 2 # Number of data loading threads

    cfg.SOLVER.IMS_PER_BATCH = 2 #Images per Batch; E.g. if we have 16 GPUs and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
    cfg.SOLVER.BASE_LR = 0.00025 #LR - learning rate
    cfg.SOLVER.MAX_ITER = 50    # number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # number of proposals to sample when training
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (leaf)

    cfg.OUTPUT_DIR="/home/imolodtsov/detectron2_trained_models/mask_rcnn_X_101_32x8d_FPN_3x"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.MODEL.DEVICE="cpu"
    predictor = DefaultPredictor(cfg)

    MetadataCatalog.get('/large_disc/pics_for_fermata/coco/leaves_test/').set(thing_classes=["leaf"])
    leaf_metadata = MetadataCatalog.get('/large_disc/pics_for_fermata/coco/leaves_test/')
    return predictor, leaf_metadata

def predict_visual(path2read,path2save, predictor, leaf_metadata, status=False):
    if status==False:
        predict_shape_only(path2read,path2save, predictor, leaf_metadata)
    else:
        predict_with_status(path2read,path2save, predictor, leaf_metadata)
        
    
def predict_shape_only(path2read,path2save, predictor, leaf_metadata):
    
    #predictor, leaf_metadata = create_predict_instance()
    
    im = cv2.imread(path2read)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=leaf_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                   
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(path2save,v.get_image()[:, :, ::-1])

def predict_with_status(path2read,path2save, predictor, leaf_metadata):
    
    #predictor, leaf_metadata = create_predict_instance()

    im = cv2.imread(path2read)
    outputs = predictor(im)

    ##Predict status
    # input: image,Instances
    # output: list of status predictions as numeric labels or text strings   

    labels=[]
    for box,mask in zip (outputs['instances'].pred_boxes,outputs['instances'].pred_masks):
        x1=max([int(box[0])-1,0])
        y1=max([int(box[1])-1,0])
        x2=min([int(box[2])+1,outputs['instances']._image_size[1]])
        y2=min([int(box[3])+1,outputs['instances']._image_size[0]])
        mod_im= cv2.bitwise_and(im, im, mask=np.asarray(mask).astype(np.uint8))

        back=np.full(mod_im.shape, 255, dtype=np.uint8)
        back = cv2.bitwise_and(back, back, mask=np.logical_not(np.asarray(mask)).astype(np.uint8))

        mod_im = cv2.bitwise_or(mod_im, back)

        crop_img = mod_im[y1:y2, x1:x2]
        labels.append(status_predict4image(crop_img))

    labels=np.asarray(labels)
    ulabels=np.unique(labels)
    clabels=[]
    ucolors=np.asarray([plt.cm.get_cmap('jet', len(ulabels))(i) for i in range(len(ulabels))])
    for label in labels:
        clabels.append(ucolors[ulabels==label][0])



    ##Visualize 
    # input: image,list of segments,list of status predictions
    # output: image

    v = Visualizer_with_labeled_colors(im[:, :, ::-1],
                   metadata=leaf_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW,   # remove the colors of unsegmented pixels
                   clabels=clabels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(path2save,v.get_image()[:, :, ::-1])


    
if __name__ == "__main__":
    create_predict_instance()
    #predict(sys.argv[1], sys.argv[2])
    #predict("/large_disc/pics_for_fermata/potato_from_VNIIF/alternarioz_a/20190729_122240.jpg","/home/imolodtsov/data/temp.jpg")
