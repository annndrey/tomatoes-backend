import math
import torch
import torchvision
import numpy as np
import os
import gc
import PIL
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict
from torchvision import transforms
import torchvision.datasets.folder as dsfolder
import matplotlib.pyplot as plt
from img_transforms import ImgResizeAndPad
from architectures import *
import torch.utils.model_zoo as model_zoo
from scipy.spatial.distance import cdist



class CompClassifier():
    def __init__(self,
                 sd_model_name, sd_model_path,
                 cl_model_name, cl_model_path, num_classes):
        self.sd_model = self._load_sd_model(sd_model_name, sd_model_path)
        self.cl_model = self._load_cl_model(cl_model_name, cl_model_path, num_classes)
        self.transform = self._get_square_transform()
        self.fnt = ImageFont.truetype('FreeMono.ttf', 60)
        self.fnt_ = ImageFont.truetype('FreeMonoBold.ttf', 20)
        
    def _load_sd_model(self, model_name, path_to_model):
        print("Loading model...")
        if model_name == 'AlexNet':
            model = AlexNet()
        elif model_name == 'DenseNet':
            model = DenseNet()
        model.cpu()
        checkpoint = torch.load(path_to_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        best_old_epoch = checkpoint['epoch']
        best_old_loss = checkpoint['loss']
        min_test_loss = best_old_loss
        print(f"Ready! Model {path_to_model}, Test loss {best_old_loss}, Saved on the {best_old_epoch} epoch")
        model.eval()
        return model
    
    def _load_cl_model(self, model_name, path_to_model, num_classes):
        """
        This function is to re-initialize function with weights saved as save_dict with current (mid Jan 2020)
        structure of storage
        :return:
        """
        model_inited=self._initialize_model(model_name, num_classes, use_pretrained=False)
        checkpoint = torch.load(path_to_model, map_location='cpu')
        #print("Saved entities: ", checkpoint.keys())
        loaded_state_dict = checkpoint['state_dict']
        self.idx_to_class = {idx:cls for idx,cls in enumerate(checkpoint['classes'])}
        model_inited.load_state_dict(loaded_state_dict)
        model_inited.eval()
        return model_inited
        
    def _initialize_model(self, model_name, num_classes,
                          feature_extract=False, use_pretrained=True):
        """
        This function is to init model of given architecture with required num_classes
        Initialize these variables which will be set in this if statement. Each of these
        variables is model specific.
        :param feature_extract: if we want to make feature extraction (not retrain all weights but rather the last
        layer
        :param use_pretrained: if to use pretrained weights or not
        :return:
        """

        model_inited = None

        if model_name == "resnet":
            """ Resnet18
            """
            model_inited = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_inited, feature_extract)
            num_ftrs = model_inited.fc.in_features
            model_inited.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "alexnet":
            """ Alexnet
            """
            model_inited = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_inited, feature_extract)
            num_ftrs = model_inited.classifier[6].in_features
            model_inited.classifier[6] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_inited = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_inited, feature_extract)
            num_ftrs = model_inited.classifier[6].in_features
            model_inited.classifier[6] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_inited = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_inited, feature_extract)
            model_inited.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_inited.num_classes = num_classes
        elif (model_name == "densenet") or (model_name == "densenet169"):
            """ Densenet
            """
            model_inited = models.densenet169(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_inited, feature_extract)
            num_ftrs = model_inited.classifier.in_features
            model_inited.classifier = nn.Linear(num_ftrs, num_classes)
        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_inited = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_inited, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_inited.AuxLogits.fc.in_features
            model_inited.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_inited.fc.in_features
            model_inited.fc = nn.Linear(num_ftrs, num_classes)
        else:
            print("Invalid model name, exiting...")
            exit()
        return model_inited
    
    def set_parameter_requires_grad(self, model_inited, feature_extracting):
        if feature_extracting:
            for param in model_inited.parameters():
                param.requires_grad = False
    
    def _get_square_transform(self):
        return {'default': transforms.Compose([
                ImgResizeAndPad(resize=224),
                transforms.ToTensor()])
                }
    
    def parse_request_picture(self, img, max_n_of_leaves):
        self.final_borders = []
        self.original = img#Image.open(fpath)
        self.dr = ImageDraw.Draw(self.original)
        zones = self._get_zones(self.original, 3, 4)
        for z in zones.keys():
            borders = (zones[z]['left'], zones[z]['top'], zones[z]['right'], zones[z]['bottom'])
            self.final_borders.append(borders)
            
        for i in range(2):
            self._cut_zones(max_n_of_leaves)
        
        res = self.classify_final_zones()
        print(["RESULTS", res])
        #display(self.original)
        #self.original.save("out.jpg")
        return res
        
    def _get_zones(self, pict, n, m):
        width, height = pict.size
        left = [width/m*i for i in range(m)]
        right = [width/m*i for i in range(1, m+1)]
        top = [height/n*i for i in range(n)]
        bottom = [height/n*i for i in range(1, n+1)]
        res = {}
        n = 1
        for l,r in zip(left, right):
            for t,b in zip(top, bottom):
                zone = {'left': int(l),
                        'top': int(t),
                        'right': int(r),
                        'bottom': int(b)}
                res[f'zone{n}'] = zone
                n += 1
        return res
    
    def _cut_zones(self, max_n_of_leaves):
        inputs = []
        zones = []
        outputs = []
        for borders in self.final_borders:
            zone = self.original.crop(borders)
            zones.append(zone)
            input = self.transform["default"](zone).unsqueeze(0)
            if len(inputs) == 0:
                inputs = input
            elif len(inputs) < 8:
                inputs = torch.cat((inputs, input), dim=0)
            else:
                outputs.append(self.sd_model.forward(inputs).detach())
                inputs = input
        
        outputs.append(self.sd_model.forward(inputs).detach())    
        outputs = torch.cat(outputs)
        
        final_borders = []
        for zone,borders,output in zip(zones,self.final_borders,outputs):
            if output.item() > max_n_of_leaves:
                small_zones = self._get_zones(zone, 2, 2)
                for z in small_zones.keys():
                    new_borders = (borders[0]+small_zones[z]['left'], 
                                   borders[1]+small_zones[z]['top'], 
                                   borders[0]+small_zones[z]['right'], 
                                   borders[1]+small_zones[z]['bottom'])
                    final_borders.append(new_borders)
            else:
                final_borders.append(borders)
        self.final_borders = final_borders
    
    def classify_final_zones(self):
        inputs = []
        outputs = []
        for borders in self.final_borders:
            zone = self.original.crop(borders)
            input = self.transform["default"](zone).unsqueeze(0)
            if len(inputs) == 0:
                inputs = input
            elif len(inputs) < 8:
                inputs = torch.cat((inputs, input), dim=0)
            else:
                outputs.append(self.cl_model.forward(inputs).detach())
                inputs = input
        
        outputs.append(self.cl_model.forward(inputs).detach())    
        outputs = torch.cat(outputs)
        outputs = [output.argmax().item() for output in outputs]
        results = [self.idx_to_class[output] for output in outputs]
        output_res = []
        for borders,res in zip(self.final_borders,results):
            if 'unhealthy' in res:
                output_res.append({"result": res, "region": borders})
                self.dr.rectangle(borders, outline = '#ff0000', width=3)
                self.dr.text((borders[0]+2, borders[1]+2), f'{res}', font=self.fnt_)
        return output_res


if __name__ == "__main__":
    sd_model_name = 'AlexNet' # 'AlexNet' or 'DenseNet'
    sd_model_path = "./AlexNet.pth"
    cl_model_name = 'resnet50'
    #cl_model_path = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cannab_10cl_izr_200_acc737.md"
    cl_model_path = "./ResNet-50_for_classification.md"
    num_classes = 5
    classifier = CompClassifier(sd_model_name=sd_model_name, sd_model_path=sd_model_path,
                                cl_model_name=cl_model_name, cl_model_path=cl_model_path, num_classes=num_classes)

    path = './cannab_4.jpeg'
    max_n_of_leaves_in_zone = 5
    
    image = classifier.parse_request_picture(path, max_n_of_leaves_in_zone)
