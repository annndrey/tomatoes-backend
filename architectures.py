import torch.nn as nn

import torchvision.models as models

class SizeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list(models.resnet34(pretrained=True).children())[:-2]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(512, 256, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(256, 16, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.Linear(16,1)]
        self.size_detection_model = nn.Sequential(*layers)

    def forward(self, x):
        return self.size_detection_model(x).squeeze(-1)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.model(x)
        output = self.activation(output)
        return output

class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet169(pretrained=True)
        self.model.classifier = nn.Linear(in_features=1664, out_features=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.model(x)
        output = self.activation(output)
        return output

class WideResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.wide_resnet50_2(pretrained=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        return output

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        return output
