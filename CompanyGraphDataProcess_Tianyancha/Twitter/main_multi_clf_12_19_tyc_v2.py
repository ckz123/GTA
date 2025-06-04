import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import logging
import time
from typing import Tuple, Dict
import numpy as np

from model12_19_tyc_v1 import MyMLP, TPLinkGNN
from dataset_tyc import ACM

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 计算相似度矩阵并应用温度系数
        similarity_matrix = torch.matmul(x, x.T) / (x.shape[0] * x.shape[0] / 16)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        
        # 移除对角线元素
        mask_self = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=x.device)
        similarity_matrix = similarity_matrix * mask_self
        
        # 构建标签掩码
        labels = y.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        
        # 计算正样本和负样本
        positive_samples = similarity_matrix * mask_pos
        negative_samples = similarity_matrix * (1 - mask_pos)
        
        # 计算损失
        pos_sum = positive_samples.sum(dim=1)
        neg_sum = negative_samples.sum(dim=1)
        
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8)).mean()
        return loss

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    def __init__(self, alpha: float = 1, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class ModelTrainer:
    """模型训练器"""
    def __init__(
        self,
        model: TPLinkGNN,
        classifier: MyMLP,
        device: torch.device,
        learning_rate: float = 1e-2,
        weight_decay: float = 5e-6
    ):
        self.model = model
        self.classifier = classifier
        self.device = device
        self.contrastive_loss = ContrastiveLoss().to(device)
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            list(model.parameters()) + list(classifier.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        
    @torch.no_grad()
    def evaluate(self, mask: torch.Tensor, phase: str = 'val') -> Tuple[float, float, float]:
        """评估模型性能"""
        self.model.eval()
        self.classifier.eval()
        
        # 获取模型输出
        x_hat_lsmp, x_hat, embd_hat = self.model()
        y_hat = self.classifier(x_hat)
        y_probs = F.softmax(y_hat, dim=1)
        
        # 获取预测结果
        predictions = torch.argmax(y_probs, dim=1)
        
        # 计算指标
        true_labels = self.model.data.y[mask].cpu().numpy()
        predicted_labels = predictions[mask].cpu().numpy()
        y_probs_np = y_probs[mask].cpu().numpy()
        
        # 计算评估指标
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average="macro")
        
        # 计算ROC-AUC
        true_labels_one_hot = F.one_hot(
            self.model.data.y[mask],
            num_classes=y_hat.size(1)
        ).cpu().numpy()
        roc_auc = roc_auc_score(
            true_labels_one_hot,
            y_probs_np,
            average="macro",
            multi_class="ovr"
        )
        
        logger.info(
            f"{phase.capitalize()} - Accuracy: {accuracy:.4f}, "
            f"F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
        )
        return accuracy, f1, roc_auc
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        self.classifier.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        x_hat_lsmp, x_hat, embd_hat = self.model()
        
        # 计算对比损失
        train_mask = self.model.data.train_mask
        contrast_loss1 = self.contrastive_loss(
            x_hat_lsmp[train_mask],
            self.model.data.y[train_mask]
        )
        contrast_loss2 = self.contrastive_loss(
            embd_hat[train_mask],
            self.model.data.y[train_mask]
        )
        
        # 分类损失
        y_hat = self.classifier(x_hat)
        cls_loss = self.criterion(
            y_hat[train_mask],
            self.model.data.y[train_mask].long()
        )
        
        # 总损失
        loss = cls_loss + contrast_loss1 + contrast_loss2
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载数据
    data = ACM()[0].to(device)
    logger.info("Data loaded successfully")
    
    # 模型参数
    in_channels = data.x.size(1)
    hidden_dim = 128
    
    # 初始化模型
    model = TPLinkGNN(
        data,
        in_channels=in_channels,
        hidden_num_lsmp=hidden_dim,
        hidden_num_tfmp=hidden_dim,
        type_link=4,
        num_layers_lsmp=2,
        num_layers_tfmp=1,
        k_topo_dim=hidden_dim
    ).to(device)
    
    classifier = MyMLP(hidden_dim, 64, 2).to(device)
    
    # 创建训练器
    trainer = ModelTrainer(model, classifier, device)
    
    # 训练循环
    num_epochs = 500
    best_val_f1 = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练
        loss = trainer.train_epoch()
        
        # 验证
        val_acc, val_f1, val_roc = trainer.evaluate(
            model.data.val_mask,
            phase='val'
        )
        
        # 测试
        test_acc, test_f1, test_roc = trainer.evaluate(
            model.data.test_mask,
            phase='test'
        )
        
        # 更新最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
        
        # 记录训练信息
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {loss:.4f} - Time: {epoch_time:.2f}s"
        )
        
    logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")

if __name__ == "__main__":
    main() 