import torch
# from dataset import Twitter
from torch_geometric.nn import GATConv
#from models import *
import random
# from utils import *
import torch.nn.functional as F
import torch.optim as optim
# import functions for model training
import random
import time
import sys
from sklearn.metrics import f1_score, roc_auc_score
import networkx as nx
from torch.optim.lr_scheduler import StepLR
# from layer import LSMP_Layers
#from layer import MyMLP
from model import MyMLP
# from dataset import Twitter
from model12_19_tyc_abla import TPLinkGNN
# from model import AttnFusioner
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
#from dataset_others import ACM, IMDB, DBLP#, PubMed
# from dataset_others_metapath import ACM, IMDB
# from dataset_others import IMDB
from dataset_tyc import ACM
# from dataset_others_distinct import ACM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
acm 3 classes
imdb 3 classes
imdb random walk need 2 times global and 2 times central
dblp 4 classes
dblp random walk need 2 times global and 2 times central?
dblp to be checked
imdb need focal loss because of imbalanced class_num

low degree need more loacl info which means high alpha(lower 1-alpha)
'''

# data=Twitter()[0]
# data.to(device)

in_channels=300
out_channels=128
type_link=2
num_layers=1

# in_channels=300
hidden_dim=128

twi=ACM()[0]
print('***********************')
print(twi.labeled_mask.sum())
print(twi.central_mask.sum())
print((twi.labeled_mask*twi.central_mask).sum())
print(max(twi.y)+1)
print((twi.y==0).sum())
print((twi.y==1).sum())
print((twi.y==2).sum())
print((twi.y==-1).sum())
print('aaaaaaaaaaaaaaaa')
print(twi.central_mask[twi.train_mask].sum())   
print(((twi.y[twi.train_mask]==-1)).sum())
in_channels=twi.x.size(1)
# print(twi)
twi=twi.to('cuda:0')
# pre_transform_1 = MyMLP(in_channels, in_channels, in_channels)
# pre_transform_2 = MyMLP(in_channels, in_channels, in_channels)
# pre_transform_3 = MyMLP(in_channels, in_channels, in_channels)
# twi.adjacency_matrix[twi.adjacency_matrix!=0]=1
tplink=TPLinkGNN(twi, in_channels=in_channels, hidden_num_lsmp=128, hidden_num_tfmp=128, 
                 type_link=4, num_layers_lsmp=2, num_layers_tfmp=1, k_topo_dim=128,)
tplink=tplink.to('cuda:0')
mymlp=MyMLP(hidden_dim, 64, 2)
mymlp=mymlp.to('cuda:0')
# pre_transform_1=pre_transform_1.to('cuda:0')
# pre_transform_2=pre_transform_2.to('cuda:0')
# pre_transform_3=pre_transform_3.to('cuda:0')
# attnFusioner=AttnFusioner(hidden_dim, hidden_dim)
tplink.train()
epoch=500
# optimizer = optim.Adam(list(pre_transform_1.parameters()) + list(pre_transform_2.parameters()) + \
#                        list(pre_transform_3.parameters()) + list(tplink.parameters()) + \
#                         list(mymlp.parameters()), lr=5e-3)
# ACM
# optimizer = optim.Adam(list(tplink.parameters()) + \
#                         list(mymlp.parameters()), lr=1e-3)
# IMDB
optimizer = optim.Adam(list(tplink.parameters()) + \
                       list(mymlp.parameters()), lr=1e-2)#, weight_decay=5e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# x1=pre_transform_1(twi.x[:5912])
# x2=pre_transform_2(twi.x[5912:5912+3025])
# x3=pre_transform_3(twi.x[5912+3025:])
# twi.x=torch.cat([x1, x2, x3], dim=0)
# twi.x[:5841]=pre_transform_1(twi.x[:5841])
# twi.x[5841:5841+4661]=pre_transform_2(twi.x[5841:5841+4661])
# twi.x[5841+4661:]=pre_transform_3(twi.x[5841+4661:])


# data=data.to('cuda:0')
def sup_contrsat_loss(x, y):
    # similarity=torch.matmul(x, x.t())
    # criterion=nn.CrossEntropyLoss()
    # # y = torch.arrange(x.size(0)).to(x.device)
    # similarity_label=torch.zeros(x.size(0),max(y)+1).to(x.device)
    # similarity[:,0]=1
    # for i in range(x.size(0)):
    #     # 获取当前样本的标签
    #     current_label = y[i]
        
    #     # 根据标签条件累加相似度
    #     same_label_sum = similarity[i][y == current_label].sum()  # 同标签相似度之和
    #     diff_label_sum = similarity[i][y != current_label].sum()  # 不同标签相似度之和
        
    #     # 填入 new_features
    #     similarity_label[i, 0] = same_label_sum  # 第一列是同标签相似度
    #     similarity_label[i, 1] = diff_label_sum  # 第二列是不同标签相似度
    # loss=criterion(similarity_label, y)
    similarity_matrix = torch.matmul(x, x.T)/(x.shape[0]*x.shape[0]/16)   # 计算相似度矩阵
    similarity_matrix = torch.exp(similarity_matrix)  # 使用指数函数来增加对比度
    similarity_matrix = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(x.device)  # 去掉对角线
    # 2. 构建标签掩码
    labels = y.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(x.device)  # 同类别为 1，异类别为 0
    #mask去掉对角线
    # mask=mask-torch.eye(mask.size(0)).to(x.device)
    # 3. 计算对比损失
    # 将相似度矩阵分为正样本和负样本对
    positive_samples = similarity_matrix * mask  # 选择同类别的相似度
    negative_samples = similarity_matrix * (1 - mask)  # 选择不同类别的相似度
    
    # 计算每个样本的对比损失
    pos_sum = positive_samples.sum(dim=1)
    neg_sum = negative_samples.sum(dim=1)
    
    # 损失计算
    loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8)).mean()
    
    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 计算概率
        pt = torch.exp(-BCE_loss)
        # 计算 Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # 根据 reduction 参数返回不同的结果
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# # 使用 FocalLoss
# criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
# loss = criterion(y_hat[tplink.data.train_mask], tplink.data.y[tplink.data.train_mask].long())
def train(tplink, mymlp, optimizer, scheduler, device):
    tplink.train()
    mymlp.train()
    # pre_transform_1.train()
    
    # pre_transform_1.train()
    # pre_transform_2.train()
    # pre_transform_3.train()

    optimizer.zero_grad()
    
    # for i in range(epoch):
    # x1=pre_transform_1(twi.x[:5841])
    # x2=pre_transform_2(twi.x[5841:5841+4661])
    # x3=pre_transform_3(twi.x[5841+4661:])
    # twi.x=torch.cat([x1, x2, x3], dim=0)
    
    # x1=pre_transform_1(twi.x[:5912])
    # x2=pre_transform_2(twi.x[5912:5912+3025])
    # x3=pre_transform_3(twi.x[5912+3025:])
    # twi.x=torch.cat([x1, x2, x3], dim=0)
    # x_hat, embd_hat, sup_contrast_loss, sup_contrast_loss4topo_encoder=tplink()
    x_hat_lsmp, x_hat, embd_hat=tplink()
    sup_contrast_loss=sup_contrsat_loss(x_hat_lsmp[tplink.data.train_mask], tplink.data.y[tplink.data.train_mask])
    sup_contrast_loss4topo_encoder=sup_contrsat_loss(embd_hat[tplink.data.train_mask], tplink.data.y[tplink.data.train_mask])
    # print(x_hat)
    # feat=torch.cat([x_hat_lsmp, x_hat], dim=1)
    print('contrast1',sup_contrast_loss)
    print('contrast2',sup_contrast_loss4topo_encoder)
    y_hat=mymlp(x_hat)
    # y_hat=mymlp(x_hat_lsmp)
    # focal loss for imdb
    # 使用 FocalLoss
    # criterion = FocalLoss(alpha=4, gamma=2, reduction='mean')
    # loss = criterion(y_hat[tplink.data.train_mask], tplink.data.y[tplink.data.train_mask].long())

    criterion=nn.CrossEntropyLoss()
    # print(torch.isnan(y_hat).sum())
    loss=criterion(y_hat[tplink.data.train_mask], tplink.data.y[tplink.data.train_mask].long())
    print('loss', loss.item())
    loss=loss #+sup_contrast_loss+sup_contrast_loss4topo_encoder
    # loss=loss+sup_contrast_loss4topo_encoder
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def validate(tplink, mymlp, metric='f1'):
    print('Validation Phase *****************************')
    with torch.no_grad():
        tplink.eval()
        mymlp.eval()

        # 获取 tplink 模型的输出
        x_hat_lsmp, x_hat, embd_hat = tplink()
        
        # 获取 mymlp 模型的预测
        y_hat = mymlp(x_hat)
        # y_hat=mymlp(x_hat_lsmp)
        # 使用 softmax 计算概率
        y_probs = torch.softmax(y_hat, dim=1)
        
        # 获取预测的类别
        predictions = torch.argmax(y_probs, dim=1)
        
        # 转换数据类型
        true_labels = tplink.data.y[tplink.data.val_mask].cpu().numpy()
        predicted_labels = predictions[tplink.data.val_mask].cpu().numpy()
        
        # 计算准确率
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # 计算 F1 分数 (macro 平均)
        f1 = f1_score(true_labels, predicted_labels, average="macro")
        
        # 计算 ROC-AUC 分数
        # 需要将 true_labels 和 y_probs 转换为 one-hot 编码
        true_labels_one_hot = torch.nn.functional.one_hot(tplink.data.y[tplink.data.val_mask], num_classes=y_hat.size(1)).cpu().numpy()
        y_probs_np = y_probs[tplink.data.val_mask].cpu().numpy()
        roc_auc = roc_auc_score(true_labels_one_hot, y_probs_np, average="macro", multi_class="ovr")
        
        print("Validation Accuracy:", accuracy)
        print("Validation F1 Score:", f1)
        print("Validation ROC-AUC:", roc_auc)
        
        return accuracy, f1, roc_auc

def test(tplink, mymlp, metric='f1'):
    print('0*****************************')
    with torch.no_grad():
        tplink.eval()
        mymlp.eval()
        # pre_transform_1.eval()
        # pre_transform_2.eval()
        # pre_transform_3.eval()
        # x1=pre_transform_1(twi.x[:5912])
        # x2=pre_transform_2(twi.x[5912:5912+3025])
        # x3=pre_transform_3(twi.x[5912+3025:])
        # twi.x=torch.cat([x1, x2, x3], dim=0)

        # 获取 tplink 模型的输出
        x_hat_lsmp, x_hat, embd_hat = tplink()
        # print('1*****************************')
        # feat=torch.cat([x_hat_lsmp, x_hat], dim=1)
        # 获取 mymlp 模型的预测
        y_hat = mymlp(x_hat)
        # y_hat = mymlp(x_hat_lsmp)
        # print("NaN values in y_hat:", torch.isnan(y_hat).sum())
        # print('2*****************************')
        
        # 使用 softmax 计算概率
        y_probs = torch.softmax(y_hat, dim=1)
        
        # 获取预测的类别
        predictions = torch.argmax(y_probs, dim=1)
        
        # 转换数据类型
        true_labels = tplink.data.y[tplink.data.test_mask].cpu().numpy()
        predicted_labels = predictions[tplink.data.test_mask].cpu().numpy()
        
        # 计算准确率
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # 计算 F1 分数 (macro 平均)
        f1 = f1_score(true_labels, predicted_labels, average="macro")
        
        # 计算 ROC-AUC 分数
        # 需要将 true_labels 和 y_probs 转换为 one-hot 编码
        true_labels_one_hot = torch.nn.functional.one_hot(tplink.data.y[tplink.data.test_mask], num_classes=y_hat.size(1)).cpu().numpy()
        y_probs_np = y_probs[tplink.data.test_mask].cpu().numpy()
        roc_auc = roc_auc_score(true_labels_one_hot, y_probs_np, average="macro", multi_class="ovr")
        
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("ROC-AUC:", roc_auc)
        
        return accuracy, f1, roc_auc
    
def train_topoembd(tplink, mymlp, optimizer, scheduler, device):
    tplink.train()
    mymlp.train()
    # for i in range(epoch):
    optimizer.zero_grad()
    x_hat, embd_hat, sup_contrast_loss, sup_contrast_loss4topo_encoder=tplink()
    # print(x_hat)
    # print(sup_contrast_loss)
    y_hat=mymlp(embd_hat)
    criterion=nn.BCELoss()
    loss=criterion(y_hat[tplink.data.train_mask], tplink.data.y[tplink.data.train_mask].view(-1, 1).float())
    print(loss.item())
    
    loss=loss+sup_contrast_loss+sup_contrast_loss4topo_encoder
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

    pass
# 查看 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 查看 CUDA 版本
print("CUDA version:", torch.version.cuda)

# 查看当前是否有可用的 GPU 设备
print("Is CUDA available:", torch.cuda.is_available())

print(tplink.data.test_mask.sum()) 
print('test', twi.y[tplink.data.test_mask].sum())


for i in range(epoch):
    # loss=train_topoembd(tplink, mymlp, optimizer, scheduler, device)
    # print(loss)
    loss=train(tplink, mymlp, optimizer, scheduler, device)
    # print('4*****************************')
    acc, f1, roc = test(tplink, mymlp)
    print('test')
    print(f'Epoch {i+1}/{epoch}, Loss {loss}, Acc {acc}, F1 Score: {f1}, Roc_auc {roc}')
    acc, f1, roc = validate(tplink, mymlp)
    print('validate')
    print(f'Epoch {i+1}/{epoch}, Acc {acc}, F1 Score: {f1}, Roc_auc {roc}')


# print('central_nodes', twi.central_mask.sum())
# print(torch.eq(twi.y[twi.central_mask],1).sum())
# print(torch.eq(twi.y[twi.central_mask],0).sum())
# print(twi.central_mask.sum())