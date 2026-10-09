
# ==========
# 1.模块引入
# ==========

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# ==========
# 2.定义读取函数
# ==========

def read_data(path):
    df=pd.read_csv(path)
    return df

# ==========
# 3.数据集划分
# ==========

def split_data(df,target_col,test_size=0.2,random_state=42):
    X=df.drop(columns=[target_col])
    y=df[target_col]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state, stratify=y)
    return X_train,X_test,y_train,y_test

# ==========
# 4.数据处理
# ==========

def prepare_data(X_train,X_test):

    # 1.处理age，填入中值

    age_median=X_train['Age'].median()
    X_train['Age']=X_train['Age'].fillna(age_median)
    X_test['Age']=X_test['Age'].fillna(age_median)

    # 2.处理embarked，填入众数

    embarked_mode= X_train['Embarked'].mode()[0]
    X_train['Embarked']=X_train['Embarked'].fillna(embarked_mode)
    X_test['Embarked']=X_test['Embarked'].fillna(embarked_mode)

    # 3.处理title

    #匹配name里的title
    X_train['Title']=X_train['Name'].str.extract(r',\s*([^\.]+)\.')[0]
    X_test['Title']=X_test['Name'].str.extract(r',\s*([^\.]+)\.')[0]
    #合并同义词
    title_map={'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'}
    X_train['Title']=X_train['Title'].replace(title_map)
    X_test['Title']=X_test['Title'].replace(title_map)
    #合并稀有词
    common_titles=['Mr','Miss','Mrs','Master']
    X_train['Title']=X_train['Title'].where(X_train['Title'].isin(common_titles),'Rare')
    X_test['Title']=X_test['Title'].where(X_test['Title'].isin(common_titles),'Rare')
    #扔掉name
    X_train = X_train.drop(columns='Name')
    X_test = X_test.drop(columns='Name')

    # 4.处理cabin

    #has_cabin
    X_train['has_cabin']=X_train['Cabin'].notna().astype(int)
    X_test['has_cabin']=X_test['Cabin'].notna().astype(int)
    #deck
    X_train['Deck']=X_train['Cabin'].str[0]
    X_test['Deck']=X_test['Cabin'].str[0]
    X_train['Deck']=X_train['Deck'].fillna('Unknown')
    X_test['Deck']=X_test['Deck'].fillna('Unknown')
    #继续合并
    common_deck=['A','B','C','D','E','F','Unknown']
    X_train['Deck']=X_train['Deck'].where(X_train['Deck'].isin(common_deck),'Rare')
    X_test['Deck']=X_test['Deck'].where(X_test['Deck'].isin(common_deck), 'Rare')
    #扔掉cabin
    X_train=X_train.drop(columns='Cabin')
    X_test=X_test.drop(columns='Cabin')

    # 5.处理ticket

    #ticket_group_size
    ticket_group_size=X_train['Ticket'].value_counts()
    X_train['ticket_group_size']=X_train['Ticket'].map(ticket_group_size)
    X_test['ticket_group_size']=X_test['Ticket'].map(ticket_group_size)
    X_test['ticket_group_size']=X_test['ticket_group_size'].fillna(1)
    #扔掉ticket
    X_train=X_train.drop(columns='Ticket')
    X_test=X_test.drop(columns='Ticket')

    # 6.处理sibsp和parch

    #family_size
    X_train['family_size']=X_train['SibSp']+X_train['Parch']+1
    X_test['family_size']=X_test['SibSp']+X_test['Parch']+1
    #is_alone
    X_train['is_alone']=(X_train['family_size']==1).astype(int)
    X_test['is_alone']=(X_test['family_size']==1).astype(int)

    # 7.删除id

    X_train=X_train.drop(columns='PassengerId')
    X_test=X_test.drop(columns='PassengerId')

    # 8.性别编码

    sex_map={'male':0,'female':1}
    X_train['Sex']=X_train['Sex'].map(sex_map)
    X_test['Sex']=X_test['Sex'].map(sex_map)

    # 9.embarked，title，deck的独热编码

    # 编码,并转整型
    categorical_cols=['Embarked','Title','Deck']
    train_categorical=pd.get_dummies(X_train[categorical_cols],dtype=int)
    test_categorical=pd.get_dummies(X_test[categorical_cols],dtype=int)
    # 对齐
    test_categorical=test_categorical.reindex(columns=train_categorical.columns,fill_value=0)
    # 拼接
    X_train=pd.concat([X_train,train_categorical],axis=1)
    X_test=pd.concat([X_test,test_categorical],axis=1)
    # 删除
    X_train=X_train.drop(columns=categorical_cols)
    X_test=X_test.drop(columns=categorical_cols)

    # 10.标准化

    # 筛出来连续数字的列
    numeric_cols=X_train.select_dtypes(include='number').columns
    numeric_cols=[col for col in numeric_cols if X_train[col].nunique()>2]
    # 标准化
    train_mean=X_train[numeric_cols].mean()
    train_std=X_train[numeric_cols].std(ddof=0)
    X_train[numeric_cols]=(X_train[numeric_cols]-train_mean)/train_std
    X_test[numeric_cols]=(X_test[numeric_cols]-train_mean)/train_std

    # 11.存数据

    preprocess_info={
        'age_median':age_median,
        'embarked_mode':embarked_mode,
        'ticket_group_size':ticket_group_size,
        'sex_map':sex_map,
        'title_map':title_map,
        'common_deck':common_deck,
        'common_titles':common_titles,
        'categorical_cols':categorical_cols,
        'one_hot_columns':train_categorical.columns.tolist(),
        'numeric_cols':numeric_cols,
        'train_mean':train_mean,
        'train_std':train_std,
        'feature_columns':X_train.columns.tolist()# 保证用户输入的能够对应
    }

    return X_train, X_test,preprocess_info

# ==========
# 5.转张量
# ==========

def to_tensor(X_train,X_test,y_train,y_test):

    #转数组
    X_train_np=X_train.to_numpy(dtype=np.float32)
    X_test_np=X_test.to_numpy(dtype=np.float32)
    y_train_np=y_train.to_numpy(dtype=np.float32)
    y_test_np=y_test.to_numpy(dtype=np.float32)
    #转张量
    X_train_tensor=torch.tensor(X_train_np)
    X_test_tensor=torch.tensor(X_test_np)
    y_train_tensor=torch.tensor(y_train_np).reshape(-1,1)
    y_test_tensor=torch.tensor(y_test_np).reshape(-1,1)

    return X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor

# ==========
# 6.dataset和dataloader
# ==========

# 定义dataset类，使之能一个个取
class TicketDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
# 定义创建dataset的函数
def create_datasets(X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor):
    train_dataset=TensorDataset(X_train_tensor,y_train_tensor)
    test_dataset=TensorDataset(X_test_tensor,y_test_tensor)
    return train_dataset,test_dataset
# 定义dataloader创建函数，使之能一批一批的取
def create_dataloaders(train_dataset,test_dataset,batch_size=32,seed=42):
    g=torch.Generator()
    g.manual_seed(seed)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,generator=g)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader





































