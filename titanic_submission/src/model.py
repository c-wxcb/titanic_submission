
# ==========
# 1.模块引入
# ==========

import torch
import torch.nn as nn

# ==========
# 2.模型构建
# ==========

class TitanicModel(nn.Module):
    def __init__(self,input_dim,hidden_dims,output_dim=1):
        super().__init__()
        layers=[]
        in_dim=input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim,hidden_dim))
            layers.append(nn.ReLU())
            in_dim=hidden_dim
        layers.append(nn.Linear(in_dim,output_dim))
        self.model=nn.Sequential(*layers)# 按顺序执行多个层，*解包，将列表里的每个元素按当作参数传进去
    def forward(self,x):
        return self.model(x)
















