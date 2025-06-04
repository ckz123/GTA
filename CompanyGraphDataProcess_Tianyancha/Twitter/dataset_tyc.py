import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.transforms import ToUndirected
import shutil
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.manifold import TSNE



x = pd.read_csv('CompanyGraphDataProcess_Tianyancha/Twitter/tyc_data/raw/company_subgraph_selected.csv')
label = pd.read_csv('CompanyGraphDataProcess_Tianyancha/Twitter/tyc_data/raw/label_subgraph.csv')
adjacency_matrix = np.load('CompanyGraphDataProcess_Tianyancha/Twitter/tyc_data/raw/adj.npy')
adj_list = np.load("CompanyGraphDataProcess_Tianyancha/Twitter/tyc_data/raw/adj_list.npy")
x = x.drop(['cid'], axis=1)
# 中值填充nan
x = x.fillna(x.median())
x = x.to_numpy()
print(x.shape)
label = label.drop(['cid'], axis=1)
label = label.to_numpy()
print(label.shape)
label = label.reshape(-1)
print(label.shape)
# ->tensor
adjacency_matrix = torch.tensor(adjacency_matrix)
adj_tensor_list = [torch.tensor(item, dtype=torch.float32) for item in adj_list]

class ACM(InMemoryDataset):
    def __init__(self, root='CompanyGraphDataProcess_Tianyancha/Twitter/tyc_data', transform=None, pre_transform=None,split='random',train_val_test_ratio=[0.4,0.2,0.4],
                 x=x,label=label,adjacency_matrix=adjacency_matrix, adj_list=adj_tensor_list):
        
        self.x=x
        self.label=label
        # self.adjacency_matrix=adjacency_matrix
        self.adjacency_matrix=torch.zeros_like(adjacency_matrix)
        # for i in range(len(adj_list)):
        #     self.adjacency_matrix = self.adjacency_matrix + adj_list[i]
        self.adjacency_matrix = adj_list[0] + adj_list[1] + adj_list[2] + adj_list[3]
        # self.adjacency_matrix = self.adjacency_matrix.numpy()
        self.adj_list=adj_list
        # for i in range(len(adj_list)):
        #     self.adj_list[i]=self.adj_list[i].numpy()

        self.split=split
        
        self.train_val_test_ratio=train_val_test_ratio
        
        super(ACM, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        #ToUndirected(merge=True)(self.data)
        if(self.split!=None):
            self.split_data()

    @property
    def raw_file_names(self):
        return ['X.npy','label.npy','adjacency_matrix.npy']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
     
    def split_data(self):
        assert self.num_classes == max(self.label) + 1
        data = self.get(0)
        label_num = len(data.y)
        print(label_num)
        #  初始化 mask
        data.train_mask = torch.BoolTensor([False] * label_num)
        data.val_mask = torch.BoolTensor([False] * label_num)
        data.test_mask = torch.BoolTensor([False] * label_num)
        # data.central_mask = torch.BoolTensor([False] * label_num)
        data.labeled_mask = torch.BoolTensor([False] * label_num)
        print(data.labeled_mask.shape)
        print(data.y.shape)
        data.y = data.y.reshape(-1)
        # 将 y 中等于 0 或 1 的样本设置为 central_mask
        data.labeled_mask[(data.y != -1)] = True

        # 对于每一个类别的标签
        for i in range(self.num_classes):
            # 只选择 y 中不等于 -1 的部分
            idx = ((data.y == i) & (data.y != -1)).nonzero(as_tuple=False).view(-1)
            num_class = len(idx)

            if num_class > 0:
                # 按比例划分训练、验证和测试集
                num_train_per_class = int(np.ceil(num_class * self.train_val_test_ratio[0]))
                num_val_per_class = int(np.floor(num_class * self.train_val_test_ratio[1]))
                num_test_per_class = num_class - num_train_per_class - num_val_per_class

                assert num_test_per_class > 0, f"Class {i} has insufficient samples for test set"

                # 随机打乱索引并划分为训练、验证和测试集
                idx_perm = torch.randperm(num_class)
                idx_train = idx[idx_perm[:num_train_per_class]]
                idx_val = idx[idx_perm[num_train_per_class:num_train_per_class + num_val_per_class]]
                idx_test = idx[idx_perm[num_train_per_class + num_val_per_class:]]

                # 设置 mask
                data.train_mask[idx_train] = True
                data.val_mask[idx_val] = True
                data.test_mask[idx_test] = True

        # 使用 collate 方法保存数据
        self.data, self.slices = self.collate([data])
        print(data.central_mask.sum())
        print(data.train_mask.sum())
        print(data.val_mask.sum())
        print(data.test_mask.sum())

    def get_similarity(self,file_path_feature,file_path_topology):
        #get the similarity matrix
        #input: file_path_feature, file_path_topology
        #output: similarity_matrix
        #similarity_matrix: the similarity matrix of the nodes
        #file_path_feature: the file path of the feature
        #file_path_topology: the file path of the topology
        #feature_matrix: the feature matrix of the nodes
        #topology_matrix: the topology matrix of the nodes
        feature_similarity_matrix = torch.load(file_path_feature)
        topology_similarity_matrix = torch.load(file_path_topology)
        #similarity_matrix = np.dot(feature_matrix,feature_matrix.T) + np.dot(topology_matrix,topology_matrix.T)
        return feature_similarity_matrix, topology_similarity_matrix
    
    def generate_central_mask(self, adj_matrix, k):
        # Step 1: 计算每个节点的度
        degrees = np.sum(adj_matrix, axis=1)  # 或者 axis=0, 对称矩阵情况下都可以

        # Step 2: 创建central_mask，度大于k的节点为True，否则为False
        central_mask = degrees > k

        return central_mask
    
    def process(self):
        x=torch.from_numpy(self.x).float()
        #tsne = TSNE(n_components=300, random_state=42)
        #x = tsne.fit_transform(x)
        # tranform=nn.Linear(x.size(1),300)
        # x=tranform(x)
        x=x[:,:300]
        y=torch.from_numpy(self.label).long()
        
        edge_index_1=torch.from_numpy(np.array(np.where(self.adj_list[0]==1))).long()
        edge_index_2=torch.from_numpy(np.array(np.where(self.adj_list[1]==1))).long()
        edge_index_3=torch.from_numpy(np.array(np.where(self.adj_list[2]==1))).long()
        edge_index_4=torch.from_numpy(np.array(np.where(self.adj_list[3]==1))).long()
        # edge_index_3=torch.from_numpy(np.array(np.where(self.adjacency_matrix==3))).long()
        # edge_index_4=torch.from_numpy(np.array(np.where(self.adjacency_matrix==4))).long()
        # edge_index_5=torch.from_numpy(np.array(np.where(self.adjacency_matrix==5))).long()
        
        edge_index_list=[edge_index_1,edge_index_2,edge_index_3,edge_index_4]
        # edge_index_list=[edge_index_1,edge_index_2,edge_index_3,edge_index_4,edge_index_5]
        edge_index=torch.from_numpy(np.array(np.where(self.adjacency_matrix>=1))).long()
        
        # adjacency_matrix_1=torch.from_numpy(np.where(self.adjacency_matrix == 1, self.adjacency_matrix, 0)).float()
        # adjacency_matrix_2=torch.from_numpy(np.where(self.adjacency_matrix == 2, self.adjacency_matrix, 0)).float()
        # # adjacency_matrix_3=torch.from_numpy(np.where(self.adjacency_matrix == 3, self.adjacency_matrix, 0)).float()
        # # adjacency_matrix_4=torch.from_numpy(np.where(self.adjacency_matrix == 4, self.adjacency_matrix, 0)).float()
        # # adjacency_matrix_5=torch.from_numpy(np.where(self.adjacency_matrix == 5, self.adjacency_matrix, 0)).float()
        # # adjacency_matrix_list=[adjacency_matrix_1,adjacency_matrix_2,adjacency_matrix_3,adjacency_matrix_4,adjacency_matrix_5]
        
        adjacency_matrix_list=self.adj_list      # [adjacency_matrix_1,adjacency_matrix_2]

        #top-k as central nodes
        central_mask = self.generate_central_mask(self.adjacency_matrix, 6)

        central_mask = torch.from_numpy(central_mask).bool()
        self.adjacency_matrix=torch.from_numpy(self.adjacency_matrix).float()

        # feature_similarity_matrix ,topology_similarity_matrix = self.get_similarity\
        #     ('CompanyGraphDataProcess_Tianyancha/Twitter/feature_similarity_matrix.pt',
        #      'CompanyGraphDataProcess_Tianyancha/Twitter/topology_similarity_matrix_filtered.pt')
        
        data=torch_geometric.data.Data(x=x,y=y,
                                        edge_index=edge_index,
                                        edge_index_list=edge_index_list,
                                        adjacency_matrix=self.adjacency_matrix,
                                        adjacency_matrix_list=adjacency_matrix_list,
                                        central_mask=central_mask,
                                        # feature_similarity_matrix=feature_similarity_matrix,
                                        # topology_similarity_matrix=topology_similarity_matrix
                                        )
        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  


acm = ACM()[0]
print(acm.edge_index.shape)