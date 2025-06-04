from builtins import NotImplementedError
from functools import reduce
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
# from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch import Tensor
# from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul, set_diag
from torch_geometric.nn.conv import MessagePassing, gat_conv, gcn_conv, sage_conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import SplineConv, GATConv, GATv2Conv, SAGEConv, GCNConv, GCN2Conv, GENConv, DeepGCNLayer, APPNP, JumpingKnowledge, GINConv
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from typing import Union, Tuple, Optional
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
#import torch_sparse
# from dataset import Twitter



device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class To_latent_space_and_MessagePassing_Layer(MessagePassing):
    def __init__(self,type_link:int, in_channels, out_channels, num_layers, hidden_num:int = 64 ,dropout=0.5, bias=True, **kwargs):
        super(To_latent_space_and_MessagePassing_Layer, self).__init__(aggr='add', **kwargs)
        self.hidden_num = hidden_num    
        self.num_layers = num_layers
        self.dropout = dropout
        self.type_link = type_link
        self.linears= nn.ModuleList()
        #set parellel layers for each type of link
        #self.layers = nn.ModuleList([nn.Linear(in_channels, in_channels) for i in range(self.type_link)])
        #self.layers=nn.Linear(in_channels, in_channels)
        # self.lin_f=nn.Sequential(
        #     nn.Linear(in_channels, hidden_num),
        #     nn.ReLU(),
        #     # nn.Dropout(dropout),
        #     nn.Linear(hidden_num, 1)
        # )
        # self.fusion=nn.Linear(in_channels, in_channels)
        self.attention_vector = nn.Parameter(torch.randn(in_channels * 2))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        

        self.to('cuda:0')
        self.reset_parameters()

    def reset_parameters(self):
        # for layer in self.layers:
        #     layer.reset_parameters()
        # self.attention_vector.data = nn.init.uniform_(self.attention_vector.data, a=-1.0, b=1.0)  # 1D 初始化
        pass
        # for layer in self.lin_f:
        #     if isinstance(layer, Linear):
        #         layer.reset_parameters()
        # self.fusion.reset_parameters()

    # def alpha_calculation(self, x, edge_index):
    #     alpha = self.lin_f(x)
    #     alpha_edge=alpha[edge_index[0]]+alpha[edge_index[1]]
    #     alpha_edge = F.leaky_relu(alpha_edge, 0.2)
    #     print("alpha_edge",alpha_edge.shape)
    #     return alpha_edge
    
    def alpha_calculation(self, x, edge_index):
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        e_ij = self.leaky_relu(torch.matmul(edge_features, self.attention_vector))
        # abla
        # e_ij全部置为1
        e_ij = torch.ones_like(e_ij)
        # print("e_ij",e_ij.shape)
        return e_ij.view(-1,1)

    def forward(self,x,edge_index):
        #x_list=[]
        #x_list_hat=[]x
        x_hat=[]
        # print(edge_index.size())
        alpha_edge=self.alpha_calculation(x, edge_index)
        
        #x_list.append(self.layers(x))
        # x_list.append(x)
        x_hat=(MessagePassing.propagate(self,edge_index=edge_index, x=x, alpha=alpha_edge))
        # x_hat=(self.fusion(x_list_hat))
        return x_hat, alpha_edge

        # return x_hat
    def message(self, x_j: Tensor, x_i: Tensor, alpha: Tensor,
            index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        a_f=alpha
        a_f = softmax(a_f, index, ptr, size_i)
        return x_j * a_f  
    


class AttnFusioner(nn.Module):
    def __init__(self, input_num=3, in_size=256, hidden=256):
        super(AttnFusioner, self).__init__()
        self.encoder_q = nn.Linear(in_size * input_num, hidden, bias=False)
        self.encoder_k = nn.Linear(in_size, hidden, bias=False)
        self.w_att = nn.Parameter(torch.FloatTensor(2 * hidden, hidden))
        self.input_num = input_num
        self.in_size = in_size
        self.hidden = hidden
        self.reset_parameters()
    
    def reset_parameters(self):
        self.encoder_q.reset_parameters()
        self.encoder_k.reset_parameters()
        nn.init.xavier_uniform_(self.w_att.data, gain=1.414)
    
    def forward(self, input_list):
        assert len(input_list) == self.input_num
        q = self.encoder_q(torch.cat(input_list, dim=1)) 
        q = q.repeat(self.input_num, 1) # input_num*N, hidden
        k_list = []
        for i in range(len(input_list)):
            k_list.append(self.encoder_k(input_list[i])) # N, hidden
        k = torch.cat(k_list, dim=0) # input_num*N, hidden
        attn_input = torch.cat([q, k], dim=1)
        attn_input = F.dropout(attn_input, 0.5, training=self.training)
        # print(self.training)
        e = F.elu(torch.matmul(attn_input, self.w_att)) # N*input_num, 1
        attention = F.softmax(e.view(self.input_num, -1, self.hidden).transpose(0, 1), dim=1) # N, input_num, hidden
        out = torch.stack(input_list, dim=1).mul(attention) # N, input_num, hidden
        out = out.sum(dim=1) # N, hidden
        out = out / self.input_num
        
        # out = torch.cat([out[:, i, :] for i in range(self.input_num)], dim=-1) # N, input_num * hidden
        # out = torch.cat(input_list, dim=1).mul(attention)
        return out

    


class LSMP_Layers(MessagePassing):
    def __init__(self, in_channels, out_channels, type_link, num_layers, hidden_num:int = 64 ,dropout=0.5, bias=True, **kwargs):
        super(LSMP_Layers, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.hidden_num = hidden_num    
        self.num_layers = num_layers
        self.dropout = dropout
        self.type_link = type_link
        self.layers=[]
        #单独计算每一种边的embedding，实例化多个To_latent_space_and_MessagePassing_Layer类
        for i in range(type_link):
            self.layers.append(To_latent_space_and_MessagePassing_Layer(
                type_link,in_channels, out_channels, num_layers)
                )
        #扩充了x
        self.semantic_weight_mp = nn.Linear(in_channels, in_channels, bias=False)
        self.semantic_bias_mp = nn.Parameter(torch.zeros(in_channels))
        self.attention_vector_mp = nn.Parameter(torch.randn(in_channels))

        
        # self.AttnFusioner_pre=AttnFusioner(input_num=type_link-4, in_size=in_channels, hidden=in_channels)
        # self.AttnFusioner_aft=AttnFusioner(input_num=4, in_size=in_channels, hidden=in_channels)
        # self.AttnFusioner_pre.to('cuda:0')
        # self.AttnFusioner_aft.to('cuda:0')
        
        # self.AttnFusioner=AttnFusioner(input_num=type_link, in_size=in_channels, hidden=in_channels)
        
        
    def reset_parameters(self):
        # for layer in layerlist.layers:
        #     layer.reset_parameters()
        
        for layer in self.layers:
            layer.reset_parameters()
        # self.AttnFusioner.reset_parameters()
        
        # self.AttnFusioner_pre.reset_parameters()
        # self.AttnFusioner_aft.reset_parameters()
        
    def alphaed_x(self, x_hat_list):
        meta_path_scores = []
        N=x_hat_list[0].size(0)
        for x_hat in x_hat_list:
            attention_input = torch.tanh(self.semantic_weight_mp(x_hat) + self.semantic_bias_mp)
            w_p = torch.matmul(attention_input, self.attention_vector_mp)  # (N,)
            meta_path_scores.append(w_p.mean())
        meta_path_scores = torch.stack(meta_path_scores)
        meta_path_weights = F.softmax(meta_path_scores, dim=0)  # (P,)
        
        # 3. 加权求和生成最终嵌入 Z
        final_embedding = torch.zeros(N, self.in_channels).to(device)
        for p in range(len(x_hat_list)):
            final_embedding = final_embedding + meta_path_weights[p] * x_hat_list[p]  # (N, hidden_dim)
        return final_embedding
        

    '''
    def forward(self,x,edge_index_list,edge_index):
        x_hat_list=[]
        for i in range(4, len(self.layers)):
            x_hat, alpha_edge = self.layers[i](x,edge_index_list[i])
            # print("x_hat---------------",x_hat.size())
            # print(x_hat.size())
            x_hat_list.append(x_hat.clone())
        # return x_hat_list
        x_hat_list=[x_hat.to(device) for x_hat in x_hat_list]
        # x_hat_list.append(x.clone())
        # x_hat=x
        # for i in range(len(x_hat_list)):
        #     x_hat=x_hat+x_hat_list[i]
        x_hat=self.AttnFusioner_pre(x_hat_list)
        # # x_hat=x_hat+x
        # x_hat=x
        x_hat_list=[]
        x_hat_two=x_hat
        # 先传递非A-X的边的特征
        for i in range(0, 4):
            x_hat_two, alpha_edge = self.layers[i](x_hat,edge_index_list[i])
            x_hat_list.append(x_hat_two.clone())
        x_hat_list=[x_hat.to(device) for x_hat in x_hat_list]
        # x_hat_list.append(x.clone())
        x_hat_two=self.AttnFusioner_aft(x_hat_list)
        return x_hat_two
    '''
    '''
    def forward(self,x,edge_index_list,edge_index):
        x_hat_list=[]
        for i in range(4,len(self.layers)):
            x_hat,alpha_edge=self.layers[i](x,edge_index_list[i])
            # print("x_hat---------------",x_hat.size())
            # print(x_hat.size())
            x_hat_list.append(x_hat.clone())
        # return x_hat_list
        # x_hat_list=[x_hat.to(device) for x_hat in x_hat_list]
        # x_hat=self.AttnFusioner(x_hat_list)
        x_hat=x_hat_list[0]
        for i in range(5,len(x_hat_list)):
            x_hat=x_hat+x_hat_list[i]
        x_hat_one=x_hat/(len(x_hat_list)-4)

        x_hat_list=[]
        for i in range(4):
            x_hat,alpha_edge=self.layers[i](x_hat_one,edge_index_list[i])
            x_hat_list.append(x_hat.clone())
        x_hat=x_hat_list[0]
        for i in range(1,4):
            x_hat=x_hat+x_hat_list[i]
        x_hat_two=x_hat/4

        x_hat=(x_hat_one+x_hat_two)/2
        return x_hat
    '''  
    
    
    # def forward(self,x,edge_index_list,edge_index):
    #     x_hat_list=[]
    #     '''
    #     for i in range(2,4):
    #     '''
    #     for i in range(2,4):
    #         x_hat,alpha_edge=self.layers[i](x,edge_index_list[i])
    #         # print("x_hat---------------",x_hat.size())
    #         # print(x_hat.size())
    #         x_hat_list.append(x_hat.clone())
    #     # return x_hat_list
    #     x_hat_list=[x_hat.to(device) for x_hat in x_hat_list]
    #     # x_hat=self.AttnFusioner(x_hat_list)
    #     for i in range(2):
    #         x_hat=x_hat+x_hat_list[i]
    #     x_hat=x_hat/2
    #     return x_hat

    def forward(self,x,edge_index_list,edge_index):
        x_hat_list=[]
        '''
        for i in range(2,4):
        '''
        for i in range(self.type_link):
            x_hat,alpha_edge=self.layers[i](x,edge_index_list[i])
            # print("x_hat---------------",x_hat.size())
            # print(x_hat.size())
            x_hat_list.append(x_hat.clone())
        # return x_hat_list
        x_hat_list=[x_hat.to(device) for x_hat in x_hat_list]
        # x_hat=self.AttnFusioner(x_hat_list)
        # for i in range(2):
        #     x_hat=x_hat+x_hat_list[i]
        
        #abla
        #直接将所有的x_hat相加/len
        x_hat=x_hat_list[0]
        for i in range(1,len(x_hat_list)):
            x_hat=x_hat+x_hat_list[i]
        x_hat=x_hat/len(x_hat_list)
        final_embedding = x_hat
        # final_embedding = self.alphaed_x(x_hat_list)
        return final_embedding

# class MyMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MyMLP, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.5)
#     def reset_parameters(self):
#         self.fc.reset_parameters()
        
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.relu(x)
#         # x = self.dropout(x)
#         return x

# class LSMP(MessagePassing):
#     def __init__(self, in_channels, out_channels, type_link, num_layers, hidden_num:int = 64 ,dropout=0.5, bias=True, **kwargs):
#         super(LSMP, self).__init__(aggr='add', **kwargs)
#         self.hidden_num = hidden_num    
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.type_link = type_link
#         self.layers=[]
#         for i in range(num_layers):
#             self.layers.append(LSMP_Layers(in_channels, out_channels, type_link, num_layers))
#             self.layers[i].to('cuda:0')
        
#     def forward(self,x,edge_index_list,edge_index):
#         x_hat=x
#         for i in range(len(self.layers)):
#             x_hat=self.layers[i](x_hat,edge_index_list,edge_index)
        
#         return x_hat

        
    
       
# class 
# twi=Twitter()
# data=twi[0]
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()
# print(device)
# data.x.to(device)
# data.edge_index_list=[data.edge_index_list[i].to(device) for i in range(len(data.edge_index_list))]
# data.edge_index=data.edge_index.to(device)


# in_channels=data.x.size(1)
# out_channels=data.x.size(1)
# type_link=3
# num_layers=2
# # layer_1=To_latent_space_and_MessagePassing_Layer(type_link,in_channels, out_channels, num_layers)
# # # layer_1=nn.DataParallel(layer_1)
# # layer_1.to(device)
# data.to(device)
# # x_hat=layer_1(data.x,data.edge_index_list,data.edge_index)
# # #print(x)
# # print(x_hat)
# epoch=300
# LSMP=LSMP_Layers(in_channels, out_channels, type_link, num_layers)
# LSMP.to(device)
# mymlp=MyMLP(in_channels, 1, 1)
# mymlp.to(device)
# for i in range(0, epoch):
    

    
#     x_hat=LSMP(data.x,data.edge_index_list,data.edge_index)
#     print(x_hat.size())
#     print(x_hat)

#     # LSMP=LSMP(in_channels, out_channels, type_link, num_layers)
#     # LSMP.to(device)
#     # x_hat=LSMP(data.x,data.edge_index_list,data.edge_index)
#     # print(x_hat)


#     x_normalized=F.normalize(data.x, p=2, dim=1)
#     similarity_matrix=torch.mm(x_normalized,x_normalized.t())
#     print(similarity_matrix.sum())
#     print('****original****')
#     #归一化feature
#     x_ha_normalized=F.normalize(x_hat, p=2, dim=1)
#     similarity_matrix=torch.mm(x_ha_normalized,x_ha_normalized.t())
#     print(similarity_matrix.sum())


    
    # y_hat=mymlp(x_hat)
    # loss=F.mse_loss(y_hat, data.y.float().unsqueeze(1))
    # print(loss)
    # loss.backward()
    # print('*************')







