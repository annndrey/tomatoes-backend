#!/usr/bin/env python
# -*- coding: utf-8 -*-


from maskrcnn_benchmark.config import cfg
from PIL import Image
import numpy as np
import predictor
import sys

def predict(path2read,
            path2save,
            mode='cpu',
            config_file = "/home/imolodtsov/maskrcnn-benchmark/finetune/models/e2e_mask_rcnn_X-101-32x8d-FPN_1x/DRAW.yaml",
            cf_threshold=0.3,
            min_image_size=500):
    
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", mode])


    coco_demo = predictor.COCODemo(
                         cfg,
                         min_image_size=min_image_size,
                         confidence_threshold=cf_threshold,
                         )
    pil_image = Image.open(path2read).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    predict_image = coco_demo.run_on_opencv_image(image)
    predict_image=Image.fromarray(predict_image[:, :, [2, 1, 0]])
    predict_image.save(path2save)


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
