import os
import numpy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torchvision.models as models
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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



model = models.__dict__['resnet50']()
model.cuda(1)

traindir = os.path.join('datasets/flowers_224x224', 'train')
train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4, pin_memory=True, sampler=None)
checkpoint = torch.load('checkpoint.pth', map_location='cuda:1')
model.load_state_dict(checkpoint['net'])



model.eval()
acc1s = []
for i, (images, target) in enumerate(train_loader):
    images = images.cuda(1, non_blocking=True)
    if torch.cuda.is_available():
        target = target.cuda(1, non_blocking=True)

    # compute output
    output = model(images)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    acc1s.append(acc1.cpu().detach().numpy())
print(numpy.mean(acc1s))
