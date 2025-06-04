#topology and features 
#topology based features passing
# from builtins import NotImplementedError
# from functools import reduce
import torch
# import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
# import numpy as np
import torch
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
# from typing import Union, Tuple
# from torch_geometric.typing import OptPairTensor, Adj, Size
from torch import Tensor
# from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul, set_diag
from torch_geometric.nn.conv import MessagePassing #, gat_conv, gcn_conv, sage_conv
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn import SplineConv, GATConv, GATv2Conv, SAGEConv, GCNConv, GCN2Conv, GENConv, DeepGCNLayer, APPNP, JumpingKnowledge, GINConv
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,OptTensor)
from typing import Optional
from torch_geometric.utils import  softmax
# import torch_sparse

# from torch_geometric.data import InMemoryDataset, Data
# from torch_geometric.utils import degree, to_networkx
# import networkx as nx
# from scipy.sparse import csr_matrix, diags
from scipy.linalg import clarkson_woodruff_transform

# from dataset import Twitter
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

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

class Topology_Encoder(nn.Module):
    def __init__(self, k: int = 128, alpha: int = 0.25, t: float = 1.0,
                 sketch_type = 'rwr', topk_similarity: int = 2048, 
                 row_topk_similarity: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.sketch_type = sketch_type
        self.alpha = alpha
        self.t = t
        self.topk_similarity = topk_similarity
        self.row_topk_similarity = row_topk_similarity
        self.lin_embd = nn.Linear(k, k, bias=False)
        self.activation = nn.ReLU()
        #self.topology_embd = self.topology_embd(sketch_type)
        # self.lin_alpha_topo=nn.Sequential(
        #     nn.Linear(k, 64),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(64, 1)
        # )
        self.sketch_type = sketch_type
    def reset_parameters(self):
        # for layer in self.lin_alpha_topo:
        #     if isinstance(layer, Linear):
        #         layer.reset_parameters()
        self.lin_embd.reset_parameters()

    def random_walk_central_node(self, adj_matrix, central_mask, labeled_mask, 
                                 k=128, alpha=0.5, t:float=1.0, x=0.5, y=1-0.5, 
                                 t1=1.0, t2=1, device='cuda'):
        t = int(1 / alpha)  # 迭代次数
        t=2
        x = 0.5  # 节点表示的平衡参数
        y = 1. - x  # 保证 x 和 y 加起来为 1

        # 将邻接矩阵和掩码移动到GPU
        #adj_matrix = adj_matrix.to_sparse()

        nnodes = adj_matrix.shape[0]
        
        # 归一化邻接矩阵
        ones_vector = torch.ones(nnodes, dtype=torch.float, device=device)
        
        degrees = adj_matrix.matmul(ones_vector)  # 每个节点的度
        degrees_inv = torch.diag(1.0 / (degrees + 1e-10))  # 度的逆矩阵

        # 构建随机游走的转移概率矩阵
        P = degrees_inv.matmul(adj_matrix)  # 归一化的转移矩阵
        M = P
        for i in range(2):
            M = (1 - alpha) * P.matmul(M) + P
        
        combined_mask = central_mask & labeled_mask
        PC=M[:, combined_mask]
        M_central = PC

        for i in range(2):  
            M_central = (1 - alpha) * P.matmul(M_central) + PC
        # for i in range(1):
        #     M = (1 - alpha) * P.matmul(M) + P
        
        # combined_mask = central_mask & labeled_mask
        # PC=M[:, combined_mask]
        # M_central = PC

        # for i in range(1):  
        #     M_central = (1 - alpha) * P.matmul(M_central) + PC
        
        cluster_sum = M.sum(axis=0).flatten()

        #不如直接来个全连接
        #可以试一试
        

        _, newcandidates = torch.topk(cluster_sum, k)
        M = M[:, newcandidates]

        # 归一化节点嵌入表示
        column_sqrt = torch.diag(1.0 / (M.sum(axis=-1) ** x + 1e-10))
        row_sqrt = torch.diag(1.0 / (M.sum(axis=0) ** y + 1e-10))
        prob = column_sqrt.matmul(M).matmul(row_sqrt)

        # 选择概率最高的节点作为中心节点索引
        _, center_idx = torch.max(prob, dim=-1)

        # 生成聚类中心矩阵
        cluster_center = torch.zeros(nnodes, k, device=device)
        cluster_center[torch.arange(nnodes), center_idx] = 1.0

        # # 随机扰动（sketching）
        random_flip = torch.diag(torch.where(torch.rand(nnodes, device=device) > 0.5, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device)))
        sketching = torch.zeros(nnodes, k, device=device)
        sketching[torch.arange(nnodes), torch.randint(0, k, (nnodes,), device=device)] = 1.0
        sketching = random_flip.matmul(sketching)

        # 计算最终的嵌入表示
        # 不能加sketching 太随机了

        ebd = adj_matrix.matmul(random_flip.matmul(cluster_center))
        ebd = adj_matrix.matmul((cluster_center))

        #是不是可以尝试分阶段的random walk
        return ebd
    
    def random_walk_GPU(self, adj_matrix, central_mask, k=128, alpha=0.5, t:float=1.0, x=0.5, y=1-0.5, t1=1.0, t2=1, device='cuda'):
        """
        基于邻接矩阵和central_mask实现随机游走节点表示，并在GPU上进行计算
        adj_matrix: 输入图的邻接矩阵（应为PyTorch张量）
        central_mask: 标识中心节点的布尔向量（长度为节点数）
        k: 嵌入的维度
        alpha: 随机游走中返回原始节点的概率
        device: 可选 'cuda' 或 'cpu'，用于控制计算是在GPU还是CPU上进行
        """

        t = int(1 / alpha)  # 迭代次数
        x = 0.5  # 节点表示的平衡参数
        y = 1. - x  # 保证 x 和 y 加起来为 1

        # 将邻接矩阵和掩码移动到GPU
        #adj_matrix = adj_matrix.to_sparse()

        nnodes = adj_matrix.shape[0]
        
        # 归一化邻接矩阵
        ones_vector = torch.ones(nnodes, dtype=torch.float, device=device)
        
        degrees = adj_matrix.matmul(ones_vector)  # 每个节点的度
        degrees_inv = torch.diag(1.0 / (degrees + 1e-10))  # 度的逆矩阵

        # 构建随机游走的转移概率矩阵
        P = degrees_inv.matmul(adj_matrix)  # 归一化的转移矩阵

        # 使用 central_mask 标识的节点作为中心节点
        central_nodes = torch.where(central_mask)[0]  # 获取被标识为中心的节点索引
        PC = P[:, central_nodes]  # 只考虑这些中心节点
        M = PC

        # 迭代 t 次，进行随机游走
        for i in range(t):
            M = (1 - alpha) * P.matmul(M) + PC

        # 根据每个中心节点的聚类得分选择候选中心节点
        cluster_sum = M.sum(axis=0).flatten()
        _, newcandidates = torch.topk(cluster_sum, k)
        M = M[:, newcandidates]

        # 归一化节点嵌入表示
        column_sqrt = torch.diag(1.0 / (M.sum(axis=-1) ** x + 1e-10))
        row_sqrt = torch.diag(1.0 / (M.sum(axis=0) ** y + 1e-10))
        prob = column_sqrt.matmul(M).matmul(row_sqrt)

        # 选择概率最高的节点作为中心节点索引
        _, center_idx = torch.max(prob, dim=-1)

        # 生成聚类中心矩阵
        cluster_center = torch.zeros(nnodes, k, device=device)
        cluster_center[torch.arange(nnodes), center_idx] = 1.0

        # 随机扰动（sketching）
        random_flip = torch.diag(torch.where(torch.rand(nnodes, device=device) > 0.5, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device)))
        sketching = torch.zeros(nnodes, k, device=device)
        sketching[torch.arange(nnodes), torch.randint(0, k, (nnodes,), device=device)] = 1.0
        sketching = random_flip.matmul(sketching)

        # 计算最终的嵌入表示
        ebd = adj_matrix.matmul(random_flip.matmul(cluster_center) + sketching)

        # ebd=self.lin_embd(ebd)
        # ebd=self.activation(ebd)

        return ebd

    def allinone(self, adjcency_matrix, embd, top_k, row_top_k):
        norm = torch.norm(embd, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1e-10), norm)
        embd = embd / norm
        similarity = embd.matmul(embd.T)
        # similarity=similarity.fill_diagonal_(0)
        topk_values, topk_indices = torch.topk(similarity.view(-1), top_k)
        # print(similarity)
        # 用num_adjacency_matrix的值替换topk_values
        '''
        游走完的embd应当局限于labeled_mask 节点中去创造连接
        而且在同类节点中是没有连接的 因为只存在metapath中了
        但是随机游走的目标也是标记节点
        '''
        # 创建一个和原始张量相同形状的零张量
        topk_tensor = torch.zeros_like(similarity)
        # 将 topk 值放入新的张量中
        topk_tensor.view(-1).scatter_(0, topk_indices, topk_values)
        # print(topk_values)
        topk_tensor=topk_tensor.view(similarity.size())
        row_topk_values, row_topk_indices = torch.topk(similarity, row_top_k, dim=1)
        # print(row_topk_values)
        # 创建一个和原始张量相同形状的零张量
        row_topk_tensor = torch.zeros_like(similarity)
        # 使用 scatter_ 来将 topm 值放入新的张量中
        row_topk_tensor.scatter_(1, row_topk_indices, row_topk_values)
        # print('row_topk_tensor')
        # print(row_topk_tensor)
        similarity_tensor=topk_tensor+row_topk_tensor
        edge_index = torch.nonzero(similarity_tensor, as_tuple=False)
        return similarity, edge_index.T
    
    def allinone_v2(self, adjcency_matrix, embd, 
                               central_mask, labeled_mask, top_k, row_top_k):
        '''转换为labeled_mask内部的连接'''
        n = adjcency_matrix.size(0)
        # embd=embd[labeled_mask]
        norm = torch.norm(embd, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1e-10), norm)
        embd = embd / norm
        similarity = embd.matmul(embd.T)
        similarity=similarity.fill_diagonal_(0)
        topk_values, topk_indices = torch.topk(similarity.view(-1), top_k)
        # print(similarity)
        # 用num_adjacency_matrix的值替换topk_values
        '''
        游走完的embd应当局限于labeled_mask 节点中去创造连接
        而且在同类节点中是没有连接的 因为只存在metapath中了
        但是随机游走的目标也是标记节点
        也可以聚合一下邻居结构信息
        '''
        # 在这先选择出labeled_mask中的节点 在这些节点中进行相似度连接
        # 然后组合原adj和现在的相似度adj mercer kernel
        

        # 创建一个和原始张量相同形状的零张量
        topk_tensor = torch.zeros_like(similarity)
        # 将 topk 值放入新的张量中
        topk_tensor.view(-1).scatter_(0, topk_indices, topk_values)
        # print(topk_values)
        topk_tensor=topk_tensor.view(similarity.size())
        row_topk_values, row_topk_indices = torch.topk(similarity, row_top_k, dim=1)
        # print(row_topk_values)
        # 创建一个和原始张量相同形状的零张量
        row_topk_tensor = torch.zeros_like(similarity)
        # 使用 scatter_ 来将 topm 值放入新的张量中
        row_topk_tensor.scatter_(1, row_topk_indices, row_topk_values)
        # print('row_topk_tensor')
        # print(row_topk_tensor)
        similarity_tensor=topk_tensor+row_topk_tensor
        # 只保留labeled_mask中的连接
        adj=adjcency_matrix
        # mask_matrix=labeled_mask.unsqueeze(0) & labeled_mask.unsqueeze(1)
        # adj[~mask_matrix] = 0
        #对adj进行归一化
        norm = torch.norm(adj, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1e-10), norm)
        adj = adj / norm
        #对similarity_tensor进行归一化
        norm = torch.norm(similarity_tensor, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1e-10), norm)
        similarity_tensor = similarity_tensor / norm
        # similarity_tensor_all = torch.zeros(n, n).to(device)
        # mask_matrix=torch.ger(labeled_mask, labeled_mask)
        # similarity_tensor_all[mask_matrix] = similarity_tensor.view(-1)

        #对adj和similarity_tensor进行mercer kernel
        # similarity_tensor = torch.matmul(adj, similarity_tensor)
        adj_kernel = (1/2)*(adj + similarity_tensor) + (adj - similarity_tensor) * (adj - similarity_tensor)
        # 12.12 23:02
        # edge_index = torch.nonzero(adj_kernel, as_tuple=False)
        edge_index = torch.nonzero(adj_kernel, as_tuple=False)
        
        return adj, edge_index.T
    
    def allinone_labeled_nodes(self, adjcency_matrix, embd, 
                               central_mask, labeled_mask, top_k, row_top_k):
        '''转换为labeled_mask内部的连接'''
        n = adjcency_matrix.size(0)
        embd=embd[labeled_mask]
        norm = torch.norm(embd, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1e-10), norm)
        embd = embd / norm
        similarity = embd.matmul(embd.T)
        similarity=similarity.fill_diagonal_(0)
        topk_values, topk_indices = torch.topk(similarity.view(-1), top_k)
        # print(similarity)
        # 用num_adjacency_matrix的值替换topk_values
        '''
        游走完的embd应当局限于labeled_mask 节点中去创造连接
        而且在同类节点中是没有连接的 因为只存在metapath中了
        但是随机游走的目标也是标记节点
        也可以聚合一下邻居结构信息
        '''
        # 在这先选择出labeled_mask中的节点 在这些节点中进行相似度连接
        # 然后组合原adj和现在的相似度adj mercer kernel
        

        # 创建一个和原始张量相同形状的零张量
        topk_tensor = torch.zeros_like(similarity)
        # 将 topk 值放入新的张量中
        topk_tensor.view(-1).scatter_(0, topk_indices, topk_values)
        # print(topk_values)
        topk_tensor=topk_tensor.view(similarity.size())
        row_topk_values, row_topk_indices = torch.topk(similarity, row_top_k, dim=1)
        # print(row_topk_values)
        # 创建一个和原始张量相同形状的零张量
        row_topk_tensor = torch.zeros_like(similarity)
        # 使用 scatter_ 来将 topm 值放入新的张量中
        row_topk_tensor.scatter_(1, row_topk_indices, row_topk_values)
        # print('row_topk_tensor')
        # print(row_topk_tensor)
        similarity_tensor=topk_tensor+row_topk_tensor
        # 只保留labeled_mask中的连接
        adj=adjcency_matrix
        mask_matrix=labeled_mask.unsqueeze(0) & labeled_mask.unsqueeze(1)
        adj[~mask_matrix] = 0
        #对adj进行归一化
        norm = torch.norm(adj, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1e-10), norm)
        adj = adj / norm
        #对similarity_tensor进行归一化
        norm = torch.norm(similarity_tensor, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1e-10), norm)
        similarity_tensor = similarity_tensor / norm
        similarity_tensor_all = torch.zeros(n, n).to(device)
        mask_matrix=torch.ger(labeled_mask, labeled_mask)
        similarity_tensor_all[mask_matrix] = similarity_tensor.view(-1)
        #对adj和similarity_tensor进行mercer kernel
        # similarity_tensor = torch.matmul(adj, similarity_tensor)
        adj_kernel = (1/2)*(adj + similarity_tensor_all) + (adj - similarity_tensor_all) * (adj - similarity_tensor_all)

        edge_index = torch.nonzero(adj_kernel, as_tuple=False)
        
        return adj_kernel, edge_index.T

    # def forward(self, adj_matrix, central_mask):
    #     if self.sketch_type == 'ori':
    #         embd = adj_matrix
    #     if self.sketch_type == 'rwr':
    #         # embd=self.random_walk(adj_matrix, central_mask, self.k)
    #         embd=self.random_walk_GPU(adj_matrix, central_mask, self.k)
    #     #return embd    
    #     if self.sketch_type == 'cwt':
    #         embd = clarkson_woodruff_transform(
    #             adj_matrix.transpose(), self.k).transepose()
        
    #     topology_similarity, topo_link_edge_index=self.allinone(
    #         embd, self.topk_similarity, self.row_topk_similarity)
        
    #     return embd, topology_similarity, topo_link_edge_index
    def forward(self, adj_matrix, central_mask, labeled_mask):
        if self.sketch_type == 'ori':
            embd = adj_matrix
        if self.sketch_type == 'rwr':
            # embd=self.random_walk(adj_matrix, central_mask, self.k)
            embd=self.random_walk_central_node(adj_matrix, central_mask, labeled_mask, self.k)
        #return embd    
        if self.sketch_type == 'cwt':
            embd = clarkson_woodruff_transform(
                adj_matrix.transpose(), self.k).transepose()
        
        # topology_similarity, topo_link_edge_index=self.allinone(
        #     adj_matrix, embd, 
        #     self.topk_similarity, self.row_topk_similarity)
        
        topology_similarity, topo_link_edge_index=self.allinone_v2(
            adj_matrix, embd, central_mask, labeled_mask, 
            self.topk_similarity, self.row_topk_similarity)
        
        # topology_similarity, topo_link_edge_index=self.allinone_labeled_nodes(
        #     adj_matrix, embd, central_mask, labeled_mask, 
        #     self.topk_similarity, self.row_topk_similarity)
        print('topo_link_edge_index')
        print(topo_link_edge_index.size())
        
        return embd, topology_similarity, topo_link_edge_index


class TFMP(MessagePassing):

    # def __init__(self, in_channels: int, out_channels: int, adj_matrix, central_mask, 
    #              num_layer: int = 2, k: int = 128, alpha: int = 0.25, t: float = 1.0,
    #              sketch_type = 'rwr', topk_similarity: int = 4096, 
    #              row_topk_similarity: int = 128, dropout: float = 0., bias: bool = True, **kwargs):
    def __init__(self, in_channels: int, out_channels: int, 
                 dropout: float = 0., bias: bool = True, **kwargs):
        super(TFMP, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.num_layers = num_layers
        # self.heads = heads
        # self.concat = concat
        # self.att_fusioner=AttnFusioner(input_num=3, in_size=in_channels, hidden=in_channels)
        self.semantic_weight_mp = nn.Linear(in_channels, in_channels, bias=False)
        self.semantic_bias_mp = nn.Parameter(torch.zeros(in_channels))
        self.attention_vector_mp = nn.Parameter(torch.randn(in_channels))
        self.attention_vector = nn.Parameter(torch.randn(in_channels * 2))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = dropout
        self.bias = bias

        # self.lin_alpha=Linear(in_channels, 1)
        
        # self.topo_encoder=Topology_Encoder(self.k, 
        #                                    self.alpha, self.t, self.sketch_type)
        # self.embd=self.topo_encoder(adj_matrix, central_mask)

        # self.topology_similarity, self.topo_link_edge_index=self.allinone(
        #     self.embd, self.topk_similarity, self.row_topk_similarity)

        self.to(device)
        
    def reset_parameters(self):
        self.lin_alpha.reset_parameters()
        # self.att_fusioner.reset_parameters()
   
    # def allinone(self, embd, top_k, row_top_k):
    #     norm = torch.norm(embd, dim=1, keepdim=True)
    #     norm = torch.where(norm == 0, torch.tensor(1e-10), norm)
    #     embd = embd / norm
    #     similarity = embd.matmul(embd.T)
    #     similarity=similarity.fill_diagonal_(0)
    #     topk_values, topk_indices = torch.topk(similarity.view(-1), top_k)
    #     print(similarity)
    #     # 创建一个和原始张量相同形状的零张量
    #     topk_tensor = torch.zeros_like(similarity)
    #     # 将 topk 值放入新的张量中
    #     topk_tensor.view(-1).scatter_(0, topk_indices, topk_values)
    #     # print(topk_values)
    #     topk_tensor=topk_tensor.view(similarity.size())
    #     row_topk_values, row_topk_indices = torch.topk(similarity, row_top_k, dim=1)
    #     # print(row_topk_values)
    #     # 创建一个和原始张量相同形状的零张量
    #     row_topk_tensor = torch.zeros_like(similarity)
    #     # 使用 scatter_ 来将 topm 值放入新的张量中
    #     row_topk_tensor.scatter_(1, row_topk_indices, row_topk_values)
    #     # print('row_topk_tensor')
    #     # print(row_topk_tensor)
    #     similarity_tensor=topk_tensor+row_topk_tensor
    #     edge_index = torch.nonzero(similarity_tensor, as_tuple=False)
    #     return similarity, edge_index.T
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
    

    def alpha_calculation(self, x, edge_index):
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        e_ij = self.leaky_relu(torch.matmul(edge_features, self.attention_vector))
        # print("e_ij",e_ij.shape)

        # abla
        e_ij = torch.zeros_like(e_ij)
        return e_ij.view(-1,1)
    

    def forward(self, x, embd, topology_similarity, topo_link_edge_index, edge_index_list):
        # def forward(self, x, embd, adj_matrix, central_mask):
        # 本应该接入topology encoder（需要恢复 而不是传参embd）
        # self.topo_encoder=Topology_Encoder(self.k, self.alpha, self.t, self.sketch_type)
        # embd=self.topo_encoder(adj_matrix, central_mask)
        # topology_similarity, topo_link_edge_index=self.allinone(embd, self.topk_similarity, self.row_topk_similarity)
        
        
        topo_link_edge_index=topo_link_edge_index.to(device)
        # by adjacency matrix weight
        # alpha_topo_link=topology_similarity[topo_link_edge_index[0],topo_link_edge_index[1]]
        # alpha_topo_link=alpha_topo_link.unsqueeze(1)
        # by attention
        alpha_topo_link=self.alpha_calculation(embd, topo_link_edge_index)
        x_hat = x
        print(topo_link_edge_index.size())
        x_hat_topo=(MessagePassing.propagate(self, 
                                            edge_index = topo_link_edge_index, x = x_hat, 
                                            alpha = alpha_topo_link))
        x_hat_list = []
        x_hat_list.append(x_hat_topo.clone())
        # 12.19 16:18
        for i in range(len(edge_index_list)):
            alpha = self.alpha_calculation(x, edge_index_list[i])
            x_hat = MessagePassing.propagate(self, 
                                             edge_index = edge_index_list[i], 
                                            x = x, alpha = alpha)
            x_hat_list.append(x_hat.clone())


        # abla 
        x_hat = x_hat_list[0]
        for i in range(1, len(x_hat_list)):
            x_hat = x_hat + x_hat_list[i]
        x_hat = x_hat / len(x_hat_list)
        # x_hat = self.alphaed_x(x_hat_list)
        return x_hat
        
    def message(self, x_j: Tensor, x_i: Tensor, alpha: Tensor,
            index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        a_f=alpha
        a_f = softmax(a_f, index, ptr, size_i)
        return x_j * a_f  

class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyMLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        # x = self.dropout(x)
        return x


