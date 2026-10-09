
# ==========
# 0.导入
# ==========

from data import(read_data,split_data,prepare_data,to_tensor,create_datasets,create_dataloaders)
from model import TitanicModel
from train import (create_training_objects,train_model)
from plot import (plot_loss,plot_accuracy)
from pathlib import Path
from datetime import datetime
import torch
import random
import numpy as np

# ==========
# 1.路径(ps:正式项目发现的问题，统一规范一下比较好)
# ==========

project_root=Path(__file__).resolve().parent.parent# 得到本文件当前位置的上一级再上一级目录，即titanic_submission
data_path=project_root/'train.csv'
doc_dir=project_root/'docs'# 得到相应路径
if True:
    run_name=datetime.now().strftime("%Y%m%d-%H%M%S")# 得到每次运行的图片名，m月份，M分钟
    run_dir=doc_dir/'runs'/run_name# 生成路径
    run_dir.mkdir(parents=True,exist_ok=True)# 创建目录，需要的话父目录一并创建，存在相应目录不要报错
    save_path=run_dir
else:
    final_dir=doc_dir/'final'
    final_dir.mkdir(parents=True,exist_ok=True)
    save_path=final_dir

# ==========
# 2.超参数
# ==========

target_col='Survived'
test_size=0.2
seed=42
torch.manual_seed(seed)# 副本里发现每次不一样，赶紧固定
batch_size=32
hidden_dims=[16,8]
output_dim=1
lr=0.001
epochs=100

# ==========
# 3.读取数据
# ==========

df=read_data(data_path)

# ==========
# 4.划分数据集
# ==========

X_train, X_test, y_train, y_test = split_data(df,target_col=target_col,test_size=test_size,random_state=seed)

# ==========
# 5.数据处理
# ==========

X_train, X_test,preprocess_info=prepare_data(X_train, X_test)

# ==========
# 6.转tensor
# ==========

X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor=to_tensor(X_train,X_test,y_train,y_test)

# ==========
# 7.转dataset
# ==========

train_dataset,test_dataset=create_datasets(X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor)

# ==========
# 8.转dataloader
# ==========

train_dataloader,test_dataloader=create_dataloaders(train_dataset,test_dataset,batch_size=batch_size,seed=seed)

# ==========
# 9.创建模型
# ==========

input_dim=X_train.shape[1]
model=TitanicModel(input_dim=input_dim,hidden_dims=hidden_dims,output_dim=output_dim)


# ==========
# 10.loss+optimizer
# ==========

loss_function,optimizer=create_training_objects(model=model,lr=lr)

# ==========
# 11.训练
# ==========

model,history,best_test_accuracy,best_test_epoch,best_state_dict=train_model(model,train_dataloader,test_dataloader,loss_function,optimizer,epochs=epochs)

# ==========
# 12.保存
# ==========

checkpoint={'model_state_dict':model.state_dict(),
            'input_dim':input_dim,
            'hidden_dims':hidden_dims,
            'output_dim':output_dim,
            'preprocess_info':preprocess_info,
            'epochs':epochs,
            'lr':lr,
            'batch_size':batch_size,
            'epoch':epochs,
            'best_test_accuracy':best_test_accuracy,
            'best_test_epoch':best_test_epoch,
            'best_state_dict':best_state_dict}
model_path=save_path/'model.pth'
torch.save(checkpoint,model_path)

# ==========
# 13.画图
# ==========

plot_loss(history['train_losses'],save_path/'train_losses.png')
plot_accuracy(history['train_accuracies'],history['test_accuracies'],save_path/'test_accuracies.png')

# ==========
# 14.结果
# ==========
# best_test_accuracy=max(history['test_accuracies'])
# best_test_epoch=history['test_accuracies'].index(best_test_accuracy)+1
# final_test_accuracy=history['test_accuracies'][-1]
# best_result_train_accuracy=history['train_accuracies'][best_test_epoch-1]
# best_result_train_loss=history['train_losses'][best_test_epoch-1]
#
# print('='*20,'最佳结果','='*20)
# print('最佳轮次:',best_test_epoch)
# print('该轮次测试集准确率：',best_test_accuracy)
# print('该轮次训练集准确率：',best_result_train_accuracy)
# print('该轮次训练损失：',best_result_train_loss)
# print('最终轮次：',checkpoint['epoch'])
# print('最终测试集准确率：',final_test_accuracy)

print('='*20,'最终结果','='*20)
print('最佳轮次',checkpoint['best_test_epoch'])
print('该轮次测试集准确率',checkpoint['best_test_accuracy'])
print('最终轮次',checkpoint['epochs'])
print('最终测试集准确率',history['test_accuracies'][-1])




