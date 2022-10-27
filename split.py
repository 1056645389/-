#遍历训练数据，将数据以8：2划分为训练集和验证集,

import os
name = [name for name in os.listdir('work/train/IMAGES') if name.endswith('.jpg')]

train_name_list=[]
for i in name:
    tmp = os.path.splitext(i)
    train_name_list.append(tmp[0])

# 构造图片-xml的链接文件ori_train.txt
with open("./work/train/ori_train.txt","w") as f:
    for i in range(len(train_name_list)):
        if i!=0: f.write('\n')
        line='IMAGES/'+train_name_list[i]+'.jpg'+" "+"ANNOTATIONS/"+train_name_list[i]+'.xml' 
        f.write(line)

# 构造label.txt
labels=['crazing','inclusion','pitted_surface','scratches','patches','rolled-in_scale']
with open("./work/train/labels.txt","w") as f:
    for i in range(len(labels)):
        line=labels[i]+'\n'
        f.write(line)

# 将ori_train.txt随机按照eval_percent分为验证集文件和训练集文件
# eval_percent 验证集所占的百分比
import random
eval_percent=0.2;

data=[]
with open("work/train/ori_train.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        data.append(line)

index=list(range(len(data)))
random.shuffle(index)

# 构造验证集文件
cut_point=int(eval_percent*len(data))
with open("./work/train/val_list.txt","w") as f:
    for i in range(cut_point):
        if i!=0: f.write('\n')
        line=data[index[i]]
        f.write(line)

# 构造训练集文件
with open("./work/train/train_list.txt","w") as f:
    for i in range(cut_point,len(data)):
        if i!=cut_point: f.write('\n')
        line=data[index[i]]
        f.write(line)