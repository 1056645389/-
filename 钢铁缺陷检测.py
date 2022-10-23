#首先将训练集解压缩
get_ipython().system('unzip -oq /home/aistudio/data/data105746/train.zip -d /home/aistudio/work/')
#测试集集解压缩
get_ipython().system('unzip -oq /home/aistudio/data/data105747/test.zip -d /home/aistudio/work/')
#删除生成的_MACOSX
get_ipython().system('rm -rf /home/aistudio/work/__MACOSX')

#遍历训练数据，将数据以8：2划分为训练集和验证集,如果已经完成了，就不需要在进行此步骤了

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


# # **2、安装需要的PaddleX版本**

# In[2]:


# 安装paddlex
# 需要注意paddlex1对于版本有所要求，所以最好更新对应的包版本
get_ipython().system('pip install "numpy<=1.19.5" -i https://mirror.baidu.com/pypi/simple')
get_ipython().system(' pip install paddlex==2.0.0')


# **引入包并设为GPU训练，如果没有GPU则使用CPU训练**

# In[3]:


#引入需要使用的库
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['GPU_VISIBLE_DEVICES'] = '0'#似乎不需要使用这条语句
import paddlex as pdx
import numpy as np


# # **3、定义数据处理流程**
# 
# 定义图像处理流程transforms,主要是进行一些数据增强的操作

# In[4]:


from paddlex import transforms
train_transforms = transforms.Compose([
    #transforms.MixupImage(mixup_epoch=250),
    #transforms.RandomDistort(),
    #transforms.RandomExpand(),
    #transforms.RandomCrop(),
    transforms.RandomResizeByShort(short_sizes=[640, 672, 704, 736, 768, 800],
                          max_size=1333,
                          interp='RANDOM'), 
    transforms.RandomHorizontalFlip(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])#在数据增强方面，大多数增强方式都不利于模型精度的提高，因此只选用了图片翻转，后期为了训练的稳定性，将关掉所有的数据增强。
#另外进行了图片的缩放和归一化便于进行训练。

eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=800, 
                    max_size=1333,
                    interp='CUBIC'), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])


# # **4、定义数据集的Dataset**
# 在训练前期仅使用训练集数据进行训练，在训练的末期将所有的图片都用于训练
# 
# 把训练集等的配置路径全部换成我们之前定义好的路径

# In[5]:


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


# # **5、定义模型网络**
# 
# 出于题目的排名是基于网络精度而进行的，所以选用精度更高的两阶段法Fast-RCNN，并且backbone选用ResNet101_vd

# In[6]:


num_classes = len(train_dataset.labels)
model = pdx.det.FasterRCNN(num_classes=num_classes,
                           backbone='ResNet101_vd')


# # **6 、定义参数的学习率以及优化方式**
# 
# 
# 因为使用了预训练模型所以在模型训练的初期使用warm-up学习率进行训练，在模型稳定了之后使用余弦退火衰减学习率。
# 
# 
# 选择带有动量的SGD作为优化器，同时对所有的参数设置了L2正则化系数。

# In[7]:


import paddle
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


# # **7、开始训练模型**
# 在训练中一共有80Epoch,而每一个Epoch均对应着一共560个step，大概一共需要训练3个小时

# In[10]:


model.train(num_epochs = num_epochs, 
            train_dataset = train_dataset, 
            train_batch_size=train_batch_size, 
            eval_dataset=eval_dataset, 
            optimizer=custom_optimizer, 
            save_interval_epochs=1, 
            log_interval_steps=2, 
            save_dir='output/T001', 
            pretrain_weights='COCO', 
            metric=None, 
            early_stop=True, 
            early_stop_patience=5, 
            use_vdl=True#,
            #pretrain_weights = None,
            #resume_checkpoint = "output/T008_101_vdMCpie3*lr/epoch_38_78.376"
            )


# # **8、进行预测**
# 
# 直接使用参考代码进行预测,并将结果写入output/T001/submission.csv的文件中提交即可！

# In[11]:


import paddlex as pdx
import os
import numpy as np
import pandas as pd

# 读取模型
model = pdx.load_model('output/T001/best_model')
#获取测试图片的序号
name = [name for name in os.listdir('work/test/IMAGES') if name.endswith('.jpg')]

test_name_list=[]
for i in name:
    tmp = os.path.splitext(i)
    test_name_list.append(tmp[0])
test_name_list.sort()
# 建立一个标号和题目要求的id的映射
num2index={'crazing':0,'inclusion':1,'pitted_surface':2,'scratches':3,'patches':4,'rolled-in_scale':5}

result_list = []

# 将置信度较好的框写入result_list
for index in test_name_list:
    image_name = 'work/test/IMAGES/'+index+'.jpg'
    predicts = model.predict(image_name)
    for predict in predicts:
        if predict['score']<0.5: continue;
        # 将bbox转化为题目中要求的格式
        tmp=predict['bbox']
        tmp[2]+=tmp[0]
        tmp[3]+=tmp[1]
        line=[index,tmp,num2index[predict['category']],predict['score']]
        result_list.append(line)

result_array = np.array(result_list)
df = pd.DataFrame(result_array,columns=['image_id','bbox','category_id','confidence'])

df.to_csv('output/T001/submission.csv',index=None)


# # **9、预测结果可视化**

# In[ ]:


for index in test_name_list:
    image_name = 'work/test/IMAGES/'+index+'.jpg'
    predicts = model.predict(image_name)
    pdx.det.visualize(image_name, predicts, threshold=0.5, save_dir='output/T001/visualize')

