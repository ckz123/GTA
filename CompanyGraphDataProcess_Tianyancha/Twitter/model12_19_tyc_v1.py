

    

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing #, gat_conv, gcn_conv, sage_conv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,OptTensor)
from typing import Optional
from torch_geometric.utils import  softmax

from scipy.linalg import clarkson_woodruff_transform
import torch.optim as optim
# from dataset import Twitter
from layer12_19_tyc_v1 import LSMP_Layers,AttnFusioner
from tfmp_v2_12_19_tyc_v1 import TFMP,Topology_Encoder


class MyMLP(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super(MyMLP, self).__init__()
        self.fc = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        self.fc.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class TPLinkGNN(nn.Module):
    def __init__(self, data, in_channels, hidden_num_lsmp:int, hidden_num_tfmp:int, 
                 type_link, num_layers_lsmp, num_layers_tfmp, k_topo_dim, 
                 activation_lsmp_func = 'relu', activation_tfmp_func = 'relu', activation_tpop_info = 'relu',
                 use_bn_lsmp=True, use_bn_tfmp=True, 
                 dropout=0.5, bias=True, **kwargs):
        super(TPLinkGNN, self).__init__()

        self.convs = nn.ModuleList()
    
        self.bns = nn.ModuleList()
        self.type_link = type_link
        self.hidden_num_lsmp = hidden_num_lsmp
        self.hidden_num_tfmp = hidden_num_tfmp
        self.num_layers_lsmp = num_layers_lsmp
        self.num_layers_tfmp = num_layers_tfmp
        self.dropout = dropout
        self.use_bn_lsmp = use_bn_lsmp
        self.use_bn_tfmp = use_bn_tfmp
        self.lin_input = nn.Linear(in_channels, hidden_num_lsmp, bias=False)
        self.data=data

        # if activation == 'relu':
        #     self.activation = nn.ReLU()
        # elif activation == 'tanh':
        #     self.activation = nn.Tanh()
        # elif activation == 'sigmoid':
        #     self.activation = nn.Sigmoid()
            
        # self.edge_dropout = edge_dropout

        self.lsmp_layer = nn.ModuleList()
        self.tfmp_layer = nn.ModuleList()

        self.bns_lsmp = nn.ModuleList()
        self.bns_tfmp = nn.ModuleList()

        self.lin_lsmp = nn.ModuleList()
        self.lin_tfmp = nn.ModuleList()

        self.activation_lsmp_func = activation_lsmp_func
        self.activation_tfmp_func = activation_tfmp_func

        self.k_topo_dim = k_topo_dim

        self.activation_lsmp = nn.ModuleList()
        for _ in range(0, num_layers_lsmp):
            self.lsmp_layer.append(LSMP_Layers(in_channels=self.hidden_num_lsmp, 
                                               out_channels=self.hidden_num_lsmp, 
                                               type_link=self.type_link, 
                                               num_layers=self.num_layers_lsmp))
            
            self.lin_lsmp.append(nn.Linear(self.hidden_num_lsmp, self.hidden_num_lsmp, bias=False))

            if(activation_lsmp_func == 'relu'):
                self.activation_lsmp.append(nn.ReLU())
            elif(activation_lsmp_func == 'Tanh'):
                self.activation_lsmp.append(nn.Tanh())
            elif(activation_lsmp_func == 'Sigmoid'):
                self.activation_lsmp.append(nn.Sigmoid())

            if(use_bn_lsmp):
                self.bns_lsmp.append(nn.BatchNorm1d(hidden_num_lsmp))
            
            # 每一层结束可以加relu和dropout
            # lsmp不需要activation的原因是使用了attn进行融合本身就是一个非线性的过程
            # 但是如果使用average的话，就需要加relu

        # 加一个loss 约束学习到的特征与原特征相似？
        self.transform = nn.Linear(self.hidden_num_lsmp, self.hidden_num_tfmp, bias=False)
        # 用原始特征还是lsmp处理后的呢？
        self.activation_between = nn.ReLU()

        self.activation_tfmp=nn.ModuleList()

        # 0.835
        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,
        #          sketch_type = 'rwr', topk_similarity = 64*4096, 
        #          row_topk_similarity = 256)
        #0.857
        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,
        #          sketch_type = 'rwr', topk_similarity = 32*4096, 
        #          row_topk_similarity = 64)
        #0.869 +tp encoding 0.8712967322312896
        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,
        #          sketch_type = 'rwr', topk_similarity = 8*4096, 
        #          row_topk_similarity = 32)
        #+tp encoding 0.885 +tp encoding without ori adj 0.883
        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,
        #          sketch_type = 'rwr', topk_similarity = 4*4096, 
        #          row_topk_similarity = 4)
        #90.6
        # imdb
        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,
        #          sketch_type = 'rwr', topk_similarity = 4*4096, 
        #          row_topk_similarity = 3)
        # tyc
        self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,   
                 sketch_type = 'rwr', topk_similarity = 4*4096, 
                 row_topk_similarity = 4)
        

        # imdb
        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,   
        #          sketch_type = 'rwr', topk_similarity = 6*4096, 
        #          row_topk_similarity = 8)
        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,   
        #          sketch_type = 'rwr', topk_similarity = 8*4096, 
        #          row_topk_similarity = 8)


        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,
        #          sketch_type = 'rwr', topk_similarity = 96*4096, 
        #          row_topk_similarity = 512)
        
        # self.topo_encoder=Topology_Encoder(k = self.k_topo_dim, alpha = 0.5, t = 1.0,
        #          sketch_type = 'rwr', topk_similarity = 16*4096, 
        #          row_topk_similarity = 128)
        
        for _ in range(0, num_layers_tfmp):
            self.tfmp_layer.append(TFMP(in_channels=hidden_num_tfmp, out_channels=hidden_num_tfmp))
            
            self.lin_tfmp.append(nn.Linear(self.hidden_num_tfmp, self.hidden_num_tfmp, bias=False))

            if(activation_tfmp_func == 'relu'):
                self.activation_tfmp.append(nn.ReLU())
            elif(activation_tfmp_func == 'Tanh'):
                self.activation_tfmp.append(nn.Tanh())
            elif(activation_tfmp_func == 'Sigmoid'):
                self.activation_tfmp.append(nn.Sigmoid())

            if(use_bn_tfmp):
                self.bns_tfmp.append(nn.BatchNorm1d(hidden_num_lsmp))
        # self.dropout=nn.Dropout(0.5)

        # 现在有的是通过结构相似性和结构信息来传递的attr 还要有单纯的topologyinfo
        self.topo_embd_transform = nn.Linear(self.k_topo_dim, self.k_topo_dim, bias=False)
        if(activation_tpop_info == 'Relu'):
            self.activation_tpop_info = nn.ReLU()
        elif(activation_tpop_info == 'Tanh'):
            self.activation_tpop_info = nn.Tanh()
        elif(activation_tpop_info == 'Sigmoid'):
            self.activation_tpop_info = nn.Sigmoid()

        self.lin_embd=nn.Linear(self.k_topo_dim, self.hidden_num_tfmp)
        self.activation_embd=nn.ReLU()

        #还需要一个concat操作 可以是attn也可以是直接连接
        #还需要一个分类器
        self.attnFusioner=AttnFusioner(input_num=2, in_size=hidden_num_lsmp, hidden=hidden_num_tfmp)

    def reset_parameters(self):
        self.lin_input.reset_parameters()
        for layer in self.lsmp_layer:
            layer.reset_parameters()
        for layer in self.tfmp_layer:
            layer.reset_parameters()
        self.transform.reset_parameters()
        self.topo_embd_transform.reset_parameters()
        for lin in self.lin_lsmp:
            lin.reset_parameters()
        for lin in self.lin_tfmp:
            lin.reset_parameters()
        
    def sup_contrsat_loss(self, x, y):
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
        similarity_matrix = torch.matmul(x, x.T) / 0.07
        similarity_matrix = torch.exp(similarity_matrix)  # 使用指数函数来增加对比度
        
        # 2. 构建标签掩码
        labels = y.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(x.device)  # 同类别为 1，异类别为 0
        
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
        
    '''
    def forward(self):

        # need data.x,data.edge_index_list,data.edge_index 
        x = self.lin_input(self.data.x)

        # x=self.data.x[:300]
        x_hat = x
        for i in range(0, self.num_layers_lsmp):
            x_hat = self.lsmp_layer[i](x_hat, self.data.edge_index_list, self.data.edge_index)
            x_hat = self.lin_lsmp[i](x_hat)

            # if(self.use_bn_lsmp):
            #     x_hat = self.bns_lsmp[i](x_hat)
            
            x_hat = self.activation_lsmp[i](x_hat)
            # x_hat = self.dropout(x_hat)
            # x_hat = F.dropout(x_hat, p=self.dropout, training=self.training)
        # x_hat = self.activation_lsmp[0](x_hat)
        x_hat_lsmp=x_hat
        #loss 有监督的对比学习实现x_hat在不同类别间的分离
        # sup_contrast_loss4lsmp=self.sup_contrsat_loss(x_hat[self.data.train_mask], self.data.y[self.data.train_mask])
        
        #transform & activation

        # x_hat = x + x_hat
        x_hat = self.transform(x_hat)
        x_hat = self.activation_between(x_hat)
        # x_hat=self.dropout(x_hat)
        # 修改了topo_encoder的输入


        embd, topology_similarity, topo_link_edge_index=self.topo_encoder(self.data.adjacency_matrix, self.data.central_mask, self.data.labeled_mask )# & self.data.train_mask)
        #需要保存哪个版本的x_hat呢？待实验
        embd_hat=self.lin_embd(embd)
        embd_hat=self.activation_embd(embd_hat)
        for i in range(0, self.num_layers_tfmp):
            x_hat = self.tfmp_layer[i](x_hat, embd_hat, topology_similarity, topo_link_edge_index)
            # x_hat = self.lin_tfmp[i](x_hat)
            # if(self.use_bn_tfmp):
            #     x_hat = self.bns_tfmp[i](x_hat)
            x_hat = self.lin_tfmp[i](x_hat)
            x_hat = self.activation_tfmp[i](x_hat)
            # x_hat = self.dropout(x_hat)
            # x_hat = x_hat + x
            # x_hat = F.dropout(x_hat, p=self.dropout, training=self.training)
        # x_hat = self.activation_tfmp[0](x_hat)
        # x_hat = x_hat + x
        
        # embd_hat=self.lin_embd(embd)
        # embd_hat=self.activation_embd(embd_hat)

        # sup_contrast_loss4topo_encoder=self.sup_contrsat_loss(embd_hat[self.data.train_mask], self.data.y[self.data.train_mask])
        
        feature_list=[x_hat, embd]

        # #放在tfmp里了
        # feature_list=[x_hat, x, embd]
        # 融合topo和topo
        x_hat = self.attnFusioner(feature_list)
        # x_hat = self.dropout(x_hat)
        
        return x_hat_lsmp, x_hat, embd_hat #, sup_contrast_loss4lsmp, sup_contrast_loss4topo_encoder
        '''
    def forward(self):

        # need data.x,data.edge_index_list,data.edge_index 
        x = self.lin_input(self.data.x)

        # x=self.data.x
        x_hat = x
        for i in range(0, self.num_layers_lsmp):
            x_hat = self.lsmp_layer[i](x_hat, self.data.edge_index_list, self.data.edge_index)
            x_hat = self.lin_lsmp[i](x_hat)

            # if(self.use_bn_lsmp):
            #     x_hat = self.bns_lsmp[i](x_hat)
            
            x_hat = self.activation_lsmp[i](x_hat)
            x_hat = F.dropout(x_hat)
            # x_hat = F.dropout(x_hat, p=self.dropout, training=self.training)
        # x_hat = self.activation_lsmp[0](x_hat)
        x_hat_lsmp=x_hat
        #loss 有监督的对比学习实现x_hat在不同类别间的分离
        # sup_contrast_loss4lsmp=self.sup_contrsat_loss(x_hat[self.data.train_mask], self.data.y[self.data.train_mask])
        
        #transform & activation

        x_hat = x + x_hat
        x_hat = self.transform(x_hat)
        x_hat = self.activation_between(x_hat)
        x_hat=F.dropout(x_hat)
        # 修改了topo_encoder的输入

        #abla
        # x_hat = x

        embd, topology_similarity, topo_link_edge_index=self.topo_encoder(self.data.adjacency_matrix, self.data.central_mask, self.data.labeled_mask )# & self.data.train_mask)
        #需要保存哪个版本的x_hat呢？待实验
        embd_hat=self.lin_embd(embd)
        embd_hat=self.activation_embd(embd_hat)
        for i in range(0, self.num_layers_tfmp):
            x_hat = self.tfmp_layer[i](x_hat, embd_hat, topology_similarity, 
                    topo_link_edge_index, self.data.edge_index_list)
            # x_hat = self.lin_tfmp[i](x_hat)
            # if(self.use_bn_tfmp):
            #     x_hat = self.bns_tfmp[i](x_hat)
            x_hat = self.lin_tfmp[i](x_hat)
            x_hat = self.activation_tfmp[i](x_hat)
            x_hat = F.dropout(x_hat)
            # x_hat = x_hat + x
            # x_hat = F.dropout(x_hat, p=self.dropout, training=self.training)
        # x_hat = self.activation_tfmp[0](x_hat)
        x_hat = x_hat + x
        
        # embd_hat=self.lin_embd(embd)
        # embd_hat=self.activation_embd(embd_hat)

        # sup_contrast_loss4topo_encoder=self.sup_contrsat_loss(embd_hat[self.data.train_mask], self.data.y[self.data.train_mask])
        
        feature_list=[x_hat, embd]
        x_hat = self.attnFusioner(feature_list)
        # x_hat = self.dropout(x_hat)
        
        return x_hat_lsmp, x_hat, embd_hat #, sup_contrast_loss4lsmp, sup_contrast_loss4topo_encoder
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # in_channels=300
# hidden_dim=128
# twi=Twitter()[0]
# twi=twi.to('cuda:0')
# tplink=TPLinkGNN(twi, in_channels=300, hidden_num_lsmp=128, hidden_num_tfmp=128, 
#                  type_link=3, num_layers_lsmp=2, num_layers_tfmp=2, k_topo_dim=128,)
# tplink=tplink.to('cuda:0')
# mymlp=MyMLP(hidden_dim, 1)
# mymlp=mymlp.to('cuda:0')


# tplink.train()
# epoch=300
# optimizer = optim.Adam(list(tplink.parameters()) + list(mymlp.parameters()), lr=5e-5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# for i in range(epoch):
#     optimizer.zero_grad()
#     x_hat, sup_contrast_loss=tplink()
#     print(x_hat)
#     print(sup_contrast_loss)
#     y_hat=mymlp(x_hat)
#     loss=F.mse_loss(y_hat, twi.y.view(-1, 1).float())
#     print(loss.item())
#     # loss=loss+sup_contrast_loss
#     loss.backward()
#     optimizer.step()
#     scheduler.step()