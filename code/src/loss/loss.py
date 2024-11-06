import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss as MAELoss
from torch.nn import *

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss
    

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        # logits: 모델의 출력값 (batch_size x num_classes)
        # targets: 정답 클래스 인덱스 (batch_size)
        
        # softmax를 적용하지 않고, nn.CrossEntropyLoss와 동일하게 동작하기 위해
        # F.cross_entropy는 logits에서 직접 loss를 계산합니다.
        loss = F.cross_entropy(logits, targets)
        return loss