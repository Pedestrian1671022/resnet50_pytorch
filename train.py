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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

model = models.__dict__['resnet50']()
torch.cuda.set_device(1)
model.cuda(1)
criterion = nn.CrossEntropyLoss().cuda(1)

optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
traindir = os.path.join('datasets/flowers_224x224', 'train')
train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4, pin_memory=True, sampler=None)
# checkpoint = torch.load('checkpoint.pth', map_location='cuda:1')
# model.load_state_dict(checkpoint['net'])


for epoch in range(0, 200):
    adjust_learning_rate(optimizer, epoch)
    # switch to train mode
    model.train()
    losses = []
    acc1s = []
    acc5s = []
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(1, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(1, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        acc1s.append(acc1.cpu().detach().numpy())
        acc5s.append(acc5.cpu().detach().numpy())
    print(numpy.mean(losses), numpy.mean(acc1s), numpy.mean(acc5s))
torch.save({'net': model.state_dict()}, 'checkpoint.pth')
