
# 初期版本，旨在跑通流程

'''
整体流程：
划分训练/验证集-->数据清洗-->特征工程-->数据处理（数值化）-->构建模型-->训练-->画图-->结果
'''

#——————————
#1.模块引入与种子固定（后续发现了问题才补加的，题目要求的固定种子一开始没理解，后续才理解）以及gpu
#——————————

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random
import matplotlib.pyplot as plt
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)# pytorch的默认随机数生成器
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False# 老实说，cudnn是啥我没了解到，目前应该也没开gpu运算，这俩没搞懂
g=torch.Generator()# pytorch的随机数生成器，一般用来弄dataloader
g.manual_seed(SEED)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


#——————————
#2.文件读入
#——————————

# print('1.开始读入')
df=pd.read_csv('train.csv')
# print(df.head(5))
# print(df.tail(5))
# print(df.shape)
# print(df.describe())
# print(df.info())
# print(df.isna().sum())  #可知age，cabin，embarked有缺失
# print('2.读入完成')

#——————————
#3.明确输入（x）输出（y）
#——————————

X=df.drop(columns='Survived')#还有X=df.drop('Survived',axis=1)可读性不好就是
Y=df['Survived']
# print(X.shape)
# print(Y.shape)
# print('3.xy完成')

#————————————
#4.划分训练集，验证集
#————————————

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=SEED,stratify=Y)

# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
# print(X_train.isa().sum())
# print('4.划分完成')

#————————————
#5.处理数据
#————————————
#5.1 age，填入中值

age_median=X_train['Age'].median()
X_train['Age']=X_train['Age'].fillna(age_median)
X_test['Age']=X_test['Age'].fillna(age_median)  #都要填入age_median防止数据泄露，毕竟想要模型预测未来的未知数据，只能从已知里学
# print("5.1Age 完成")

#5.2 embarked，填入众数

embarked_mode= X_train['Embarked'].mode()[0]
X_train['Embarked']=X_train['Embarked'].fillna(embarked_mode)
X_test['Embarked']=X_test['Embarked'].fillna(embarked_mode)  #同理
# print("5.2Embarked 完成")

#5.3 name,构造title新特征

# titles=X_train['Name'].str.extract(r',\s*([^\.]+)\.')[0]
# print(titles.head(5))# 继承了原始数据的索引
X_train['Title']=X_train['Name'].str.extract(r',\s*([^\.]+)\.')[0]# 继承原始数据索引，方便对齐，如X_train['Title']=titles
X_test['Title']=X_test['Name'].str.extract(r',\s*([^\.]+)\.')[0]# 不用填训练集数据，因为这不是从测试集里学出来的
'''
关于正则表达式
,匹配逗号
\\s,匹配空格，*，匹配0，1，2，3。。。次
（）为捕获组，即是要提取的
[]为匹配字符的
^为非，排除
\\.,点字符
+，至少一次
抓取默认返回dataframe，所以用[0]（第一列）列名提取称谓

'''

# print('5.3.1抓取完成')
# print(X_train['Title'].value_counts())# 有点多，而且有的只在训练集里有，考虑合并
# print(X_test['Title'].value_counts())
title_map={'Mlle':'Miss',
           'Ms':'Miss',
           'Mme':'Mrs'}
X_train['Title']=X_train['Title'].replace(title_map)# replace未匹配时保留原值，map则将未匹配的替换为nan
X_test['Title']=X_test['Title'].replace(title_map)
# print('5.3.2替换完成')
print(X_train['Title'].value_counts())
print(X_test['Title'].value_counts())
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']#这几个之外的大部分都是只有一两个的
X_train['Title']=X_train['Title'].where(X_train['Title'].isin(common_titles),'Rare')# 满足条件就保留，否则替换
X_test['Title']=X_test['Title'].where(X_test['Title'].isin(common_titles),'Rare')# 这个要从测试集里弄，因为是新规则
# print(X_train['Title'].value_counts())
# print(X_test['Title'].value_counts())
X_train=X_train.drop(columns='Name')
X_test=X_test.drop(columns='Name')

#5.4 cabin，构造deck和has_cabin新特征

# print(X_train['Cabin'].isnull().mean())
# print(X_test['Cabin'].isnull().mean())
X_train['Deck']=X_train['Cabin'].str[0]
X_test['Deck']=X_test['Cabin'].str[0]
X_train['Deck']=X_train['Deck'].fillna('Unknown')
X_test['Deck']=X_test['Deck'].fillna('Unknown')# 这玩意也能当个特征
# print(X_train["Deck"].value_counts())
# print(X_test["Deck"].value_counts())#继续合并
common_deck=['A', 'B', 'C', 'D', 'E', 'F','Unknown']
X_train['Deck']=X_train['Deck'].where(X_train['Deck'].isin(common_deck),'Rare')
X_test['Deck']=X_test['Deck'].where(X_test['Deck'].isin(common_deck), 'Rare')
X_train['has_cabin']=X_train['Cabin'].notna().astype(int)# 有值就是1，没值就是0
X_test['has_cabin']=X_test['Cabin'].notna().astype(int)
# print(X_train['has_cabin'].value_counts())
# print(X_test['has_cabin'].value_counts())
X_train = X_train.drop(columns=['Cabin'])
X_test = X_test.drop(columns=['Cabin'])

#5.5 ticket，构造ticket_group_size新特征

# print(X_train['Ticket'].head())
# print(X_test['Ticket'].head())
# print(X_train['Ticket'].value_counts())
# print(X_test['Ticket'].value_counts())# 有多人共用一张船票
use_all_data=False# 题干更新过一次，好像只能false
if use_all_data:
    ticket_group_size=df['Ticket'].value_counts()
else:
    ticket_group_size=X_train['Ticket'].value_counts()
X_train['ticket_group_size']=X_train['Ticket'].map(ticket_group_size)
X_test['ticket_group_size']=X_test['Ticket'].map(ticket_group_size)# 这里不用replace是要转换数值
X_train=X_train.drop(columns=['Ticket'])
X_test=X_test.drop(columns=['Ticket'])
X_test['ticket_group_size']=X_test['ticket_group_size'].fillna(1)# 防止false时，测试集中出现训练集没有的票号导致被标记成nan
'''
这里有取舍了，因为一张票可能四个人，三个分到训练集，一个分到测试集
那么到底是严格的取每张票究竟多少人，还是严格的按照不泄露测试集的数据来，是需要取舍的
前者真实，符合直觉，但是会有数据泄露，我们希望避免
后者不泄露数据，但总归有点不真实，一张4个人的票可能被弄成只有3个人
我目前是这么想的，都试试，都看看效果
ps:题目更新了，只能false
'''

#5.6 SibSp,Parch，构造familysize,isalone新特征

X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1
X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch'] + 1
X_train['IsAlone'] = (X_train['FamilySize'] == 1).astype(int)
X_test['IsAlone'] = (X_test['FamilySize'] == 1).astype(int)

#5.6 passengerid

X_train=X_train.drop(columns=['PassengerId'])
X_test=X_test.drop(columns=['PassengerId'])

#5.7 数据转换

# print(X_train.columns)
# print(X_test.columns)# 检查类别
# print(X_train.isna().sum())
# print(X_test.isna().sum())# 看看是不是没有缺失

#5.7.1 sex（0或1）

sex_map={'male':0,'female':1}
X_train['Sex']=X_train['Sex'].map(sex_map)
X_test['Sex']=X_test['Sex'].map(sex_map)
# print(X_train['Sex'])# 目前已经是数字的有pclass，age，sibsp，parch，fare，has_cabin,ticket_group_size,familysize,isalone
#还有embarked,title,deck是字符串

#5.7.2 字符类型独热编码（embarked,title,deck）

categorical_cols=['Embarked','Title','Deck']
train_categorical=pd.get_dummies(X_train[categorical_cols])
test_categorical=pd.get_dummies(X_test[categorical_cols])# 二者编码可能不一样，可能有些字符类型只有测试集有而训练集没有，也有反过来的，索引还要对齐
test_categorical=test_categorical.reindex(columns=train_categorical.columns,fill_value=0)# 如果出现测试集有而训练集没有的数据的话，会被抹掉，但是前面已经统一了，不会出现这种情况，fill_values如果不填，会弄成nan
X_train=X_train.drop(columns=categorical_cols)
X_test=X_test.drop(columns=categorical_cols)
X_train=pd.concat([X_train,train_categorical],axis=1)# 0维行，1维列，3维表格，4维表格的表格
X_test=pd.concat([X_test,test_categorical],axis=1)
# print(X_train.head())
# print(X_train.shape)
# print(X_test.head())
# print(X_test.shape)
# print(X_train.dtypes)
# print(X_train.isna().sum())
# print(X_test.isna().sum())# 有bool，转换
bool_cols=X_train.select_dtypes(include='bool').columns
X_train[bool_cols]=X_train[bool_cols].astype(int)
X_test[bool_cols]=X_test[bool_cols].astype(int)

#5.8 标准化（下面演示sklearn里的standardscaler的大体原理）

numerical_cols=X_train.select_dtypes(include='number').columns
numerical_cols=[col for col in numerical_cols if X_train[col].nunique()>2]# 区别于unique显示不同的值，会显示有多少个不同的值# 这里报错了，只写了col（列名，字符串）
# print(numerical_cols)# 拿到连续数据的列名
train_mean=X_train[numerical_cols].mean()
train_std=X_train[numerical_cols].std(ddof=0)# 默认是1，但是sklearn里默认0，统一比较好
X_train[numerical_cols]=(X_train[numerical_cols]-train_mean)/train_std
X_test[numerical_cols]=(X_test[numerical_cols]-train_mean)/train_std

#——————————
#6.模型
#——————————

#6.1 dataframe转数组再转张量

X_train_np=X_train.to_numpy(dtype=np.float32)# pytorch默认float32，防止报错
X_test_np=X_test.to_numpy(dtype=np.float32)
y_train_np=Y_train.to_numpy(dtype=np.float32)
y_test_np=Y_test.to_numpy(dtype=np.float32)# 数组作为中间量，方便做一些运算，比如此处的float32
X_train_tensor=torch.tensor(X_train_np)# 还有from_numpy,共享内存，不用拷贝，但是会关联改动，一改都变，还必须数组才能用
X_test_tensor=torch.tensor(X_test_np)# tensor就是新建了个副本
y_train_tensor=torch.tensor(y_train_np).reshape(-1,1)
y_test_tensor=torch.tensor(y_test_np).reshape(-1,1)# 最后出来的output是一‘列’，所以把这里也改成一‘列’，方便对应
# print(type(X_train_np))
# print(X_train_np.shape)
# print(type(X_train_tensor))
# print(X_train_tensor.shape)
# print(X_train_tensor.dtype)
# print(X_train_tensor)

#6.2 dataset

class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):# index的缩写，简单些
        return self.X[idx], self.y[idx]
train_dataset=TitanicDataset(X_train_tensor,y_train_tensor)
test_dataset=TitanicDataset(X_test_tensor,y_test_tensor)
# print(len(train_dataset))# 魔术技巧，用len会自动调用__len__,[x]会自动调用__getitem__
# print(len(test_dataset))
# print(train_dataset[0])

#6.3 dataloader

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,generator=g)# 一次拿32个数据，打乱
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
# print(len(train_loader))# 为23，多少个batch，区别于train_dataset的长度（样本数量）
# for x,y in train_loader:
#     print(x.shape,y.shape)
#     break

#6.4 神经网络

#6.4.1 构建输出

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(26,16)
        self.fc2=nn.Linear(16,8)
        self.fc3=nn.Linear(8,4)
        self.fc4=nn.Linear(4,2)
        self.fc5=nn.Linear(2,1)
    def forward(self,x):
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        x=torch.relu(x)
        x=self.fc3(x)
        x=torch.relu(x)
        x=self.fc4(x)
        x=torch.relu(x)
        x=self.fc5(x)# 保留原始输出，不用relu，前面用relu是因为简单，后面不用sigmod压缩是因为后面的bcewithlogitsloss里面有了
        return x# 返回logit，原始数值，在sigmoid里，负数为概率小于0.5，正数大于0.5
'''
以下部分注释主要是看单个epoch的单个batch怎么搞
最后for遍历
'''
# model=TitanicModel()
# X_batch,y_batch=next(iter(train_loader))
# output=model(X_batch)
# print(X_batch.shape)
# print(y_batch.shape)
# print(output.shape)
# print(output)

#6.4.2 计算损失

# loss_function=nn.BCEWithLogitsLoss()
# loss=loss_function(output,y_batch)
# print(loss.item())

#6.4.3 更新参数

# print(model.named_parameters())
# for param in model.named_parameters():
#     print(param)
# optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
# optimizer.zero_grad()
# loss.backward()
# print(model.fc1.weight.grad)
# optimizer.step()
# print(loss.item())

#6.4.4 综合遍历

model=TitanicModel().to(device)# 实例化
loss_function=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)# 创建优化器，相当于梯度下降
num_epochs=1000# 训练次数
train_losses=[]
train_accuracies=[]
test_accuracies=[]
for epoch in range(num_epochs):
    #——————————
    #训练
    #——————————
    model.train()# 训练模式
    epoch_loss=0.0
    correct=0
    total=0
    for X_batch,y_batch in train_loader:
        X_batch=X_batch.to(device)
        y_batch=y_batch.to(device)
        output=model(X_batch)#整个输出
        loss=loss_function(output,y_batch)# 计算损失，得到张量，所以后面给了item
        optimizer.zero_grad()# 清空梯度，默认是累加
        loss.backward()# 计算梯度
        optimizer.step()# 更新梯度
        epoch_loss = epoch_loss + loss# 累加损失，便于计算平均损失
        probability=torch.sigmoid(output)# 将logit弄成概率
        prediction=(probability>=0.5).int()# 预测出来的概率由阈值区分，布尔转整形
        correct=correct+(prediction==y_batch).sum().item()# 因为y_batch存在导致出来的是张量，所以item
        total=total+y_batch.size(0)# 总体个数，一般32，最后一个是8
    epoch_loss=epoch_loss/len(train_loader)# 平均损失
    epoch_loss=epoch_loss.item()# 张量转普通数值,前面还要loss.forword，所以不能先item
    train_accuracy=correct/total# 准确率
    # print(f'epoch {epoch+1} loss: {epoch_loss:4f} train_accuracy: {train_accuracy:4f}')
    #——————————
    #测试
    #——————————
    model.eval()# 测试模式
    correct=0
    total=0
    with torch.no_grad():# 由此开始，不记录梯度（pytorch默认记录），with，临时改变某种环境，而后自动回复
        for X_batch,y_batch in test_loader:
            X_batch=X_batch.to(device)
            y_batch=y_batch.to(device)
            output=model(X_batch)# 整个输出
            probability=torch.sigmoid(output)# 压缩logit
            prediction=(probability>=0.5).int()# 变为0，1,这俩都是tensor
            correct=correct+(prediction==y_batch).sum().item()# 正确个数
            total=total+y_batch.size(0)# 总体
        test_accuracy=correct/total# 准确率
    # print(f'epoch:{epoch+1} loss: {epoch_loss:4f} train_accuracy:{train_accuracy:4f} test_accuracy: {test_accuracy:4f}')
    #——————————
    #保存结果
    #——————————
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    if epoch in [0,9,99,199,299,399,499,599,999]:
        print(f'epoch:{epoch+1} loss:{epoch_loss} train_accuracy:{train_accuracy:6f} test_accuracy:{test_accuracy:6f}')
        '''这时候没弄种子，导致随机性很大，跑了几次，结果都不一样
        还有一次卡在了61，问了问可能是直接全给预测成0（61恰好是0在整个数据里的占比）
        所以需要固定种子
        目前随机性来源：
        训练集测试集划分（已固定）
        模型参数初始化（权重与偏置）
        dataloader的shuffle=true
        后续可能的随机数
        具体设置放在了前面
        '''
#——————————
#7.画图
#——————————

plt.figure()
# plt.plot(range(1,1001,50),train_losses,label='train_loss')
plt.plot(range(1,1001),train_accuracies,label='train_accuracy')
plt.plot(range(1,1001),test_accuracies,label='test_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig('train_accuracy.png')
plt.show()
# plt.figure()
# plt.plot(range(1,1001,50),test_accuracies,label='test_accuracy')
# plt.xlabel('epoch')
# plt.ylabel('test_accuracy')
# plt.legend()
# plt.grid()






















