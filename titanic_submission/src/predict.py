
# ===========
# 1.导入
# ===========

from pathlib import Path
import torch
import pandas as pd
from model import TitanicModel

# ==========
# 2.定义加载函数
# ==========

def load_run(run_dir):
    run_dir=Path(run_dir)
    model_path=run_dir/'model.pth'
    checkpoint=torch.load(model_path,weights_only=False)# 涉及到序列化保存和反序列化保存，这个留待学习
    model=TitanicModel(input_dim=checkpoint['input_dim'],hidden_dims=checkpoint['hidden_dims'],output_dim=checkpoint['output_dim'])
    model.load_state_dict(checkpoint['best_state_dict'])
    model.eval()
    preprocess_info=checkpoint['preprocess_info']
    return model,preprocess_info

# ===========
# 3.处理新乘客
# ===========

def prepare_new_passenger(df,preprocess_info):
    df=df.copy()
    #age
    df['Age']=df['Age'].fillna(preprocess_info['age_median'])
    #embarked
    df['Embarked']=df['Embarked'].fillna(preprocess_info['embarked_mode'])
    #title
    df['Title']=df['Name'].str.extract(r',\s*([^\.]+)\.')[0]
    df['Title']=df['Title'].replace(preprocess_info['title_map'])
    df['Title']=df['Title'].where(df['Title'].isin(preprocess_info['common_titles']),'Rare')
    #cabin
    df['has_cabin']=df['Cabin'].notna().astype(int)
    df['Deck']=df['Cabin'].str[0]
    df['Deck']=df['Deck'].fillna('Unknown')
    df['Deck']=df['Deck'].where(df['Deck'].isin(preprocess_info['common_deck']),'Rare')
    #ticket
    df['ticket_group_size']=df['Ticket'].map(preprocess_info['ticket_group_size'])
    df['ticket_group_size']=df['ticket_group_size'].fillna(1)
    #family_size
    df['family_size']=df['SibSp']+df['Parch']+1
    #is_alone
    df['is_alone']=(df['family_size']==1).astype(int)
    #sex
    df['Sex']=df['Sex'].map(preprocess_info['sex_map'])
    #onehot
    categorical_cols=preprocess_info['categorical_cols']
    categorical=pd.get_dummies(df[categorical_cols],dtype=int)
    categorical=categorical.reindex(columns=preprocess_info['one_hot_columns'],fill_value=0)
    df=pd.concat([df,categorical],axis=1)
    #drop
    delete_cols = ['PassengerId','Name','Cabin'] + categorical_cols
    df = df.drop(columns=delete_cols)
    #标准化
    numeric_cols=preprocess_info['numeric_cols']
    train_mean=preprocess_info['train_mean']
    train_std=preprocess_info['train_std']
    for col in numeric_cols:
        df[col]=(df[col]-train_mean[col])/train_std[col]
    #特征序列对齐
    df=df.reindex(columns=preprocess_info['feature_columns'])
    #转tensor
    new_passenger_numpy=df.to_numpy()
    new_passenger_tensor=torch.tensor(new_passenger_numpy,dtype=torch.float32)
    return new_passenger_tensor

# ==========
# 4.预测函数
# ==========

def predict_new_passenger(model,new_passenger_tensor):
    with torch.no_grad():
        output=model(new_passenger_tensor)
        probability=torch.sigmoid(output)
        prediction=(probability>=0.5).int()
    return prediction.item(),probability.item()










