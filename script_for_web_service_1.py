#python3
#get appropriate pytorch on this page: https://pytorch.org/

import torch
import torchvision
from torchvision import datasets, models, transforms

from collections import OrderedDict

import os

tomat_or_not_path = "C:\\work\\pomidory\\trained_models\\vgg19_deep_epochs15_batchsize20_chinese_tomat_or_not_acc9947_state_dict.md"
plant_health_or_not_path = "C:\\work\\pomidory\\trained_models\\vgg19_deep_epochs15_batchsize20_chinese_plant_health_or_not_acc995_state_dict.md"
tomat_health_or_not_path = "C:\\work\\pomidory\\trained_models\\vgg19_deep_epochs15_batchsize20_chinese_tomat_health_or_not_acc993_state_dict.md"
#plant_health_or_not_path = "/mnt/large_disk/science/robot_sel_hoz/codes/trained_models/vgg19_deep_epochs15_batchsize20_chinese_plant_health_or_not_acc995_state_dict.md"
#tomat_or_not_path = "/mnt/large_disk/science/robot_sel_hoz/codes/trained_models/vgg19_deep_epochs15_batchsize20_chinese_tomat_or_not_acc9947_state_dict.md"
#tomat_health_or_not_path = "/mnt/large_disk/science/robot_sel_hoz/codes/trained_models/vgg19_deep_epochs15_batchsize20_chinese_tomat_health_or_not_acc993_state_dict.md"


def load_pretrained_weights(model, weight_path):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
#    checkpoint = torch.load(weight_path)
    checkpoint = torch.load(weight_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        warnings.warn('The pretrained weights "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print("** The following layers are discarded due to unmatched keys or layer size: {}".format(discarded_layers))

using_model_name = 'vgg19'
num_classes_used = 2

tomat_or_not_model = torchvision.models.__dict__[using_model_name](num_classes=num_classes_used)
load_pretrained_weights(tomat_or_not_model, tomat_or_not_path)
tomat_or_not_model.eval()

plant_health_or_not_model = torchvision.models.__dict__[using_model_name](num_classes=num_classes_used)
load_pretrained_weights(plant_health_or_not_model, plant_health_or_not_path)
plant_health_or_not_model.eval()

tomat_health_or_not_model = torchvision.models.__dict__[using_model_name](num_classes=num_classes_used)
load_pretrained_weights(tomat_health_or_not_model, tomat_health_or_not_path)
tomat_health_or_not_model.eval()

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

#AL I didnt corrected it yet, now the pictures we are processing must be in any subfolder(-s) in data_dir.
#AL Ill correct it later or you can try doing it yourself
#data_dir = '/mnt/large_disk/science/robot_sel_hoz/test_pics_examples'
data_dir = 'C:\\work\\pomidory\\test_pics'

resize = (224,224)
using_data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_dataset = ImageFolderWithPaths(data_dir, using_data_transform) 
test_dataloader = torch.utils.data.DataLoader(test_dataset)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = (tomat_or_not_model, tomat_health_or_not_model, plant_health_or_not_model)
modres = ( { 0 : "Not a Tomat", 1 : "Tomat" },
           { 0 : "TomatHealthy", 1 : "TomatNonHealthy" },
           { 0 : "PlantHealthy", 1 : "PlantNonHealthy" }
)

result = {}

for model in models:
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels, fn) in enumerate(test_dataloader):
            fname = fn[0]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            key = os.path.basename(str(fname))
            if key in result:
                result[key].append( int(preds) )
            else:
                result[key]= [int(preds)]

for fn, res in result.items():
    print (fn , modres[0][res[0]], modres[1][res[1]], modres[2][res[2]], )









