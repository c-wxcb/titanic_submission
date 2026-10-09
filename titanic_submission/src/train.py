
# ==========
# 1.模块引入
# ==========

import torch
import torch.nn as nn
from sympy.stats.rv import probability
from torch import optim


# ==========
# 2.训练框架
# ==========

# 定义损失函数与优化器（清空与更新参数用的）
def create_training_objects(model,lr=0.001):
    loss_function=nn.BCEWithLogitsLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    return loss_function,optimizer
# 定义一个轮回的训练
def train_one_epoch(model,train_loader,loss_function,optimizer):
    model.train()
    total_loss=0
    correct=0
    total=0
    for X_batch,y_batch in train_loader:
        output=model(X_batch)
        loss=loss_function(output,y_batch)
        total_loss=total_loss+loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        probability=torch.sigmoid(output)
        prediction=(probability>=0.5).int()
        correct=correct+(prediction==y_batch).sum().item()
        total=total+y_batch.size(0)
    average_loss=total_loss/len(train_loader)# 每轮损失已经平均过了，所以只要在平均一次len就可以了
    accuracy=correct/total
    return average_loss,accuracy
# 定义测试函数
def evaluate_one_epoch(model,test_loader):
    model.eval()
    correct=0
    total=0
    for X_batch,y_batch in test_loader:
        output=model(X_batch)
        probability=torch.sigmoid(output)
        prediction=(probability>=0.5).int()
        correct=correct+(prediction==y_batch).sum().item()
        total=total+y_batch.size(0)
    accuracy=correct/total
    return accuracy
# 定义总体训练遍历函数
def train_model(model,train_loader,test_loader,loss_function,optimizer,epochs=100):
    train_losses=[]
    train_accuracies=[]
    test_accuracies=[]
    best_test_accuracy=0
    best_test_epoch=0
    best_state_dict={}
    for epoch in range(epochs):
        train_loss,train_accuracy=train_one_epoch(model,train_loader,loss_function,optimizer)
        test_accuracy=evaluate_one_epoch(model,test_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy=test_accuracy
            best_test_epoch=epoch+1
            for key,value in model.state_dict().items():
                best_state_dict[key]=value.clone()
        if (epoch+1)%10==0:
            print(f'epoch:{epoch+1} loss:{train_loss:.4f} train_accuracy:{train_accuracy:.4f} test_accuracy:{test_accuracy:.4f}')
    history={'train_losses':train_losses,'train_accuracies':train_accuracies,'test_accuracies':test_accuracies}
    
    return model,history,best_test_accuracy,best_test_epoch,best_state_dict





















