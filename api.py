import logging
import torch
import torchvision
import io
import numpy as np 
import json

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.logger import logger as fastapi_logger
from logging.handlers import RotatingFileHandler
from starlette.requests import Request
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image
from imgaug import augmenters as iaa 


#MODEL = "/home/annndrey/cityfarmer/cityfarm_56cl_50_acc948.md"
#MODEL = "/home/annndrey/cityfarmer/cityfarm_56cl_50_acc952_100320.md"
#MODEL = "/home/annndrey/cityfarmer/cityfarm_58cl_50_acc951_100320.md"
#MODEL = "/home/annndrey/cityfarmer/cityfarm_58cl_100_acc952_120320.md"
#MODEL = "/home/annndrey/cityfarmer/cityfarm_63cl_75_acc951_150320.md"
#MODEL = "/home/annndrey/cityfarmer/cityfarm_63cl_110_acc955_160320.md"
#MODEL = "/home/annndrey/cityfarmer/cityf_75cl_acc953_270420.md"
#MODEL = "/home/annndrey/cityfarmer/cityf_75cl_80_acc951_270420.md"
#MODEL = "/home/annndrey/cityfarmer/cityf_75cl_130_acc956_280420.md"
#MODEL = "/home/annndrey/cityfarmer/cityf_75cl_150_acc955_280420.md"
#MODEL = "/home/annndrey/cityfarmer/mestkorn_73cl_50_acc961_160520.md"
MODEL = "/home/annndrey/cityfarmer/mestkorn_73cl_100_acc965_180520.md"
USING_MODEL_NAME = 'densenet169'
#NUM_CLASSES_USED = 41
#NUM_CLASSES_USED = 63
#NUM_CLASSES_USED = 75
NUM_CLASSES_USED = 73
RESIZE = 224

formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s", "%Y-%m-%d %H:%M:%S")
handler = RotatingFileHandler('api.log', backupCount=0)
logging.getLogger().setLevel(logging.INFO)
fastapi_logger.addHandler(handler)
handler.setFormatter(formatter)
fastapi_logger.info('****************** Starting Server *****************')

app = FastAPI(redoc_url=None, docs_url=None)


class ImgResizeAndPad:
    #max_cropped_part - maximum percentage of each side length cropped (with probability ~0.5)
#max_ratio_change - maximum relative increase of the short side toward square size (with probability 0.5). Set it to 1 to keep the aspect ratio of the img always.
    def __init__(self, resize=224, max_ratio_change = 1.2):
        self.max_ratio_change = max_ratio_change
        self.resize = resize
        self.pad = iaa.size.PadToFixedSize(width = resize, height = resize, pad_mode='constant', pad_cval=0, position='center')
    def __call__(self, img):
        img = np.array(img)
        (h,w) = img.shape[0:2]
        keep_h=False
        M = w
        m = h
        if h>w :
            keep_h=True
            M = h
            m = w
        if self.max_ratio_change > 1:
            new_m = int(m*self.max_ratio_change*self.resize/M)
            if new_m>self.resize: new_m=self.resize
        else:
            new_m = int(m*self.resize/M)
        if keep_h:
            resize_to_square = iaa.Resize({"height": self.resize, "width": new_m})
        else:
            resize_to_square = iaa.Resize({"height": new_m, "width": self.resize})
        img = resize_to_square.augment_image( img )
        img = self.pad.augment_image( img )
        return img


only_make_square_transform = transforms.Compose([
    ImgResizeAndPad(resize=RESIZE),
    lambda x: Image.fromarray(x),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])


def get_model_results(model, results, img, allowed_values=False, verbose=False):
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    outputs = model(img)
    
    _, preds = torch.max(outputs, 1)
    detected_img_type = idx_to_class[int(preds)]
    fastapi_logger.debug(idx_to_class)
    fastapi_logger.debug(outputs)
    keys = model.class_to_idx.keys()
    values = [float(o) for o in outputs[-1]]
    full_dict = dict(zip(keys, values))
    fastapi_logger.debug(full_dict)
    
    if verbose:
        detected_img_type = full_dict
        detected_img_type = dict(sorted(detected_img_type.items(), key=lambda kv: kv[1], reverse=True))
    
    elif allowed_values:
        fastapi_logger.debug("FILTER")
        filter_keys = []
        for k in allowed_values:
            if k in full_dict.keys():
                filter_keys.append(k)
                
        filtered_dict = {key: full_dict[key] for key in filter_keys}
        fastapi_logger.debug("FILTER2")
        fastapi_logger.debug(filtered_dict)
        fastapi_logger.debug("FILTER3")
        if filtered_dict:
            detected_img_type = max(filtered_dict, key=lambda key: filtered_dict[key])
        else:
            detected_img_type = {}
        fastapi_logger.debug("FILTER4")
    return detected_img_type



def load_model(file_name):
    checkpoint = torch.load(file_name, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    arch = checkpoint['arch']
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    model = torchvision.models.__dict__[arch](num_classes=num_classes)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.class_to_idx = class_to_idx

    return model


@app.on_event("startup")
async def startup_event():
    global aimodel
    aimodel = ""
    trained_model = load_model(MODEL)
    trained_model.eval()
    aimodel = trained_model
    fastapi_logger.info('****************** Loading model *****************')
    print(MODEL)

    
@app.post("/loadimage")
async def create_upload_file(request: Request):
    fastapi_logger.debug("Upload file")
    form = await request.form()
    fastapi_logger.debug("Form data")
    verbose = form.get('verbose', False)
    allowed_values = form.get('allowed_values', False)
    if allowed_values:
        try:
            allowed_values = json.loads(allowed_values)
        except:
            allowed_values = False
    imagefile = form["imagefile"].filename
    contents = await form["imagefile"].read()
    img_pil = Image.open(io.BytesIO(contents))
    fastapi_logger.debug("Image read")
    img_tensor = only_make_square_transform(img_pil)
    img_tensor.unsqueeze_(0)
    img_variable = img_tensor
    fastapi_logger.debug("Image transform")
    result = {}
    result = get_model_results(aimodel, result, img_variable, allowed_values, verbose)
    fastapi_logger.debug("Image recognize")
    fastapi_logger.debug(result)
    
    return {"filename": imagefile, "length": len(contents), 'objtype': result}
