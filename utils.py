import pandas as pd

class Imagenet_Explainer(object):
    def __init__(self):
        self.path = "./01_mnist/data/imagenet_classes.csv"
        self.classes = pd.read_csv(self.path)

    def explain(self,output):
        pred_class = []
        for out in output:
            pred_class.append(torch.argmax(out).numpy())
        return pd.DataFrame(pred_class, columns=['class']).set_index('class').join(imagenet_classes, how='left')



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


import torch
import torch.nn as nn
import numpy as np
from typing import Any

class SimpleNet(nn.Module):

    def __init__(self,num_classes: int=200) -> None:
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*6*6, 1152),
            nn.ReLU(inplace=True),
            nn.Linear(1152,num_classes),
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x



a = torch.tensor(np.ones(shape=(1,3,64,64)),dtype=torch.float)
a = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)(a)
a = nn.ReLU(inplace=True)(a)
a = nn.MaxPool2d(kernel_size=3,stride=3)(a)
a = nn.AdaptiveAvgPool2d((6,6))(a)