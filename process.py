#引入需要使用的库
import matplotlib
matplotlib.use('Agg') 
import os
import paddlex as pdx
import numpy as np
import paddle

from paddlex import transforms
train_transforms = transforms.Compose([
    transforms.RandomResizeByShort(short_sizes=[640, 672, 704, 736, 768, 800],
                          max_size=1333,
                          interp='RANDOM'), 
    transforms.RandomHorizontalFlip(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=800, 
                    max_size=1333,
                    interp='CUBIC'), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])

train_dataset = pdx.datasets.VOCDetection(
    data_dir='work/train',
    file_list='work/train/train_list.txt',
    label_list='work/train/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='work/train',
    file_list='work/train/val_list.txt',
    label_list='work/train/labels.txt',
    transforms=eval_transforms)

num_classes = len(train_dataset.labels)
model = pdx.det.FasterRCNN(num_classes=num_classes,
                           backbone='ResNet101_vd')

train_batch_size = 2
num_steps_each_epoch = 1120 // train_batch_size
num_epochs = 80
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=.0075, 
    T_max=num_steps_each_epoch * 12 // 3
    )
warmup_epoch = 1
warmup_steps = warmup_epoch * num_steps_each_epoch
scheduler = paddle.optimizer.lr.LinearWarmup(
    learning_rate=scheduler,
    warmup_steps=warmup_steps,
    start_lr=0.00075,
    end_lr=.0075)
custom_optimizer = paddle.optimizer.Momentum(
            scheduler,
            momentum=.9,
            weight_decay=paddle.regularizer.L2Decay(coeff=1e-04),
            parameters=model.net.parameters())
