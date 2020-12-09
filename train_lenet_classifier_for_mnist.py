import os
import numpy as np
import torch
import torchvision
import argparse
from torchvision.models.alexnet import AlexNet
from PIL import Image
from torchvision import transforms
from torchvision import datasets
import pandas as pd
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt




# # pd.DataFrame(imagenet_classes).T.to_csv("./01_mnist/data/imagenet_classes.csv")
# model = torchvision.models.alexnet(pretrained=True).to(device="cpu")
# test_img = Image.open("./01_mnist/data/n02814533_0.JPEG")
#
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# input_tensor = preprocess(test_img)
# input_batch = input_tensor.unsqueeze(0)
#
# with torch.no_grad():
#     output = model(input_batch)
#
# print(output[0])
# print(torch.nn.functional.softmax(output[0], dim=0))
#
# explainer = Imagenet_Explainer()
# explainer.explain(output)


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


def main():

    # model = torchvision.models.alexnet(pretrained=False).to(device="cuda")
    model = SimpleNet().to(device='cuda')

    params = list(list(model.parameters()))

    train_dataset = datasets.ImageFolder(
        root="F:\\02_DATASETS\\tiny_imagenet\\tiny-imagenet-200\\tiny-imagenet-200\\train",
        transform=transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor() # ,
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss().cuda(0)

    epoch = 0

    for epoch in tqdm(range(epoch,80)):

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True, sampler=None)

        for i, (img, tgt) in enumerate(train_loader):

            images = img.to("cuda")
            target = tgt.to("cuda")
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss

            if i%10 == 0:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



if __name__== '__main__':
    main()



















