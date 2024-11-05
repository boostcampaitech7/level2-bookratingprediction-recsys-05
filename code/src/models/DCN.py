import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, MLP_Base


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ParameterList([
            nn.Parameter(torch.empty((input_dim, 1))) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.empty((input_dim,))) for _ in range(num_layers)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for w in self.w:
            nn.init.xavier_uniform_(w)
        for b in self.b:
            nn.init.constant_(b, 0.0)  # 바이어스를 상수 0으로 초기화

    def forward(self, x: torch.Tensor):
        x0 = x  # 초기 입력 x0를 저장
        for i in range(self.num_layers):
            # 수식: x_{l+1} = x0 * (x_l^T * w_l) + b_l + x_l
            x_l_w = torch.matmul(x, self.w[i])  # x_l^T * w_l 계산 (스칼라 값)
            x = x0 * x_l_w + self.b[i] + x      # 스칼라곱과 바이어스 및 x 추가
        return x


# Crossnetwork 결과를 MLP layer에 넣어 최종결과를 도출합니다.
class DeepCrossNetwork(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, args.cross_layer_num)
        self.mlp = MLP_Base(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)
        self.cd_linear = nn.Linear(args.mlp_dims[-1], 1, bias=False)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)


    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        p = self.cd_linear(x_out)
        return p.squeeze(1)
import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, MLP_Base
import numpy as np

# cross product transformation을 구현합니다.
class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.empty((input_dim,))) for _ in range(num_layers)
        ])

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m, 0)


    # def forward(self, x: torch.Tensor):
    #     x0 = x
    #     for i in range(self.num_layers):
    #         xw = self.w[i](x)
    #         x = x0 * xw + self.b[i] + x
    #     return x
    

    def forward(self, x: torch.Tensor):
        x0 = x
        for i in range(self.num_layers):
            xtranspose = torch.transpose(x, 0, 1)
            x = x0 @ xtranspose @ torch.Tensor(self.w[i]) + torch.Tensor(self.b[i]) + x
        return x


# Crossnetwork 결과를 MLP layer에 넣어 최종결과를 도출합니다.
class DeepCrossNetwork(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, args.cross_layer_num)
        self.mlp = MLP_Base(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)
        self.cd_linear = nn.Linear(args.mlp_dims[-1], 1, bias=False)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)


    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        p = self.cd_linear(x_out)
        return p.squeeze(1)
