
# ==========
# 1.模块引入
# ==========

import matplotlib.p as plt

# ==========
# 2.损失图像函数
# ==========

def plot_loss(train_losses,save_path):
    epochs=range(1,len(train_losses)+1)
    plt.figure()
    plt.plot(epochs,train_losses,label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.title('Training Loss Curve')
    plt.savefig(save_path)
    plt.show()

# ==========
# 3.准确率图像
# ==========

def plot_accuracy(train_accuracies,test_accuracies,save_path):
    epochs=range(1,len(train_accuracies)+1)
    plt.figure()
    plt.plot(epochs,train_accuracies,label='Train Accuracy')
    plt.plot(epochs,test_accuracies,label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.title('Train & Test Accuracy Curve')
    plt.savefig(save_path)
    plt.show()



