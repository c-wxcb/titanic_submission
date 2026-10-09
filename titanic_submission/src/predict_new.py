
# ==========
# 1.引入预测函数
# ==========

from predict import (load_run,
                     predict_new_passenger,
                     prepare_new_passenger)
from pathlib import Path
import pandas as pd

# ==========
# 2.路径
# ==========

project_dir=Path(__file__).resolve().parent.parent
run_dir=(project_dir/'docs'/'runs'/'20261009-175904')

# ==========
# 3.新乘客
# ==========

new_passenger=pd.DataFrame(
[{
    "PassengerId": 1001,
    "Pclass": 3,
    "Name": "Smith,Mr.Mary",
    "Sex": "male",
    "Age": 29,
    "SibSp": 4,
    "Parch": 2,
    "Ticket": "NEW 001",
    "Fare": 80.0,
    "Cabin": "B85",
    "Embarked": "C"
}]
)

# ==========
# 4.预测
# ==========

model,process_info=load_run(run_dir)
new_passenger=prepare_new_passenger(new_passenger,process_info)
prediction,probability=predict_new_passenger(model,new_passenger)
print(prediction)
print(probability)

# ==========
# 5.关于准确率卡在0.55的排查
# ==========

# prepare_before=pd.DataFrame([{
#     "PassengerId": 1001,
#     "Pclass": 2,
#     "Name": "Smith,Miss.Mary",
#     "Sex": "female",
#     "Age": 2,
#     "SibSp": 2,
#     "Parch": 2,
#     "Ticket": "NEW 001",
#     "Fare": 80.0,
#     "Cabin": "B85",
#     "Embarked": "C"
# }])
# prepare_after=pd.DataFrame([{
#     "PassengerId": 1001,
#     "Pclass": 2,
#     "Name": "Smith,Miss.Mary",
#     "Sex": "male",
#     "Age": 2,
#     "SibSp": 2,
#     "Parch": 2,
#     "Ticket": "NEW 001",
#     "Fare": 80.0,
#     "Cabin": "B85",
#     "Embarked": "C"
# }])
#

# # 两份原始乘客数据分别进行预处理
# X_before=prepare_new_passenger(prepare_before, process_info)
# X_after=prepare_new_passenger(prepare_after, process_info)
# # 逐层比较两个输入经过模型后的结果
# a=X_before.clone()
# b=X_after.clone()
# with torch.no_grad():
#     for i,layer in enumerate(model.model):# enumerate使得遍历能够带标号返回，也就是i，model.model就是model里面的网络结构属性
#         a=layer(a)
#         b=layer(b)# 按当前层计算两份数据
#         max_diff=(a-b).abs().max().item()# 最大差距，abs为绝对值
#         print(
#             f"第{i}层 {type(layer).__name__}: "
#             f"是否相同={torch.equal(a, b)}, "
#             f"最大差异={max_diff:.10f}"
#         )




















