import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, MLP_Base

class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ParameterList([
            nn.Parameter(torch.empty((input_dim, input_dim))) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.empty((input_dim,))) for _ in range(num_layers)
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        for w in self.w:
            nn.init.xavier_uniform_(w)
        for b in self.b:
            nn.init.constant_(b, 0.0)  # 바이어스를 0으로 초기화

    def forward(self, x: torch.Tensor):
        x0 = x  # 초기 입력 x0를 저장
        for i in range(self.num_layers):
            # 수식: x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l
            wx = torch.matmul(x, self.w[i])  # W_l x_l 계산
            x = x0 * (wx + self.b[i]) + x    # 원소별 곱 적용 및 갱신
        return x


# DCN V2의 DeepCrossNetwork 클래스 (stacked 및 parallel 모드 지원)
class DeepCrossNetworkV2(nn.Module):
    def __init__(self, args, data, mode='stacked'):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mode = mode  # 'stacked' 또는 'parallel' 모드를 선택

        # Cross Network 및 MLP 초기화
        self.cn = CrossNetworkV2(self.embed_output_dim, args.cross_layer_num)
        self.mlp = MLP_Base(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)

        # parallel 모드에 필요한 선형 결합 계층
        if self.mode == 'parallel':
            self.linear_cn = nn.Linear(self.embed_output_dim, args.mlp_dims[-1], bias=False)
            self.linear_mlp = nn.Linear(args.mlp_dims[-1], args.mlp_dims[-1], bias=False)

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
        # 임베딩을 통해 각 필드 임베딩 후 펼치기
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)

        # mode에 따라 stacked 또는 parallel 구조를 사용하여 출력 계산
        if self.mode == 'stacked':
            # Stacked: Cross Network의 출력을 MLP의 입력으로 연결
            x_l1 = self.cn(embed_x)
            x_out = self.mlp(x_l1)
        elif self.mode == 'parallel':
            # Parallel: Cross Network와 MLP 출력을 결합
            x_cn = self.cn(embed_x)
            x_mlp = self.mlp(embed_x)
            x_out = self.linear_cn(x_cn) + self.linear_mlp(x_mlp)

        # 최종 출력 계산
        p = self.cd_linear(x_out)
        return p.squeeze(1)