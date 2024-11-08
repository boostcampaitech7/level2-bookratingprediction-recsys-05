import torch
import torch.nn as nn
import numpy as np
from ._helpers import FeaturesEmbedding, CNN_Base

class MultiHeadFeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim * num_heads)
    
    def forward(self, x):
        # x: (batch_size, num_fields)
        embed_x = self.embedding(x)  # shape: (batch_size, num_fields, embed_dim * num_heads)
        # Reshape to (batch_size, num_fields, num_heads, embed_dim)
        batch_size, num_fields, total_embed_dim = embed_x.size()
        embed_x = embed_x.view(batch_size, num_fields, self.num_heads, self.embed_dim)
        # Permute to (batch_size, num_heads, num_fields, embed_dim)
        embed_x = embed_x.permute(0, 2, 1, 3)
        # Flatten num_fields and embed_dim dimensions
        embed_x = embed_x.reshape(batch_size, self.num_heads, -1)  # (batch_size, num_heads, num_fields * embed_dim)
        return embed_x


class ExponentialCrossNetworkV3(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads=1, layer_norm=True, batch_norm=False, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.ws = nn.ModuleList()
        self.bs = nn.ParameterList()
        for i in range(num_layers):
            self.ws.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.bs.append(nn.Parameter(torch.zeros(input_dim)))
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(input_dim // 2))
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))
        self.masker = nn.ReLU()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch_size, num_heads, input_dim)
        batch_size, num_heads, input_dim = x.size()
        x = x.view(batch_size * num_heads, input_dim)  # (batch_size * num_heads, input_dim)
        for i in range(self.num_layers):
            H = self.ws[i](x)  # shape: (batch_size, num_heads, input_dim // 2)
            if len(self.batch_norms) > i:
                H = self.batch_norms[i](H)
            if len(self.layer_norms) > i:
                H_norm = self.layer_norms[i](H)
                mask = self.masker(H_norm)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)  # (batch_size, num_heads, input_dim)
            x = x * (H + self.bs[i]) + x
            if len(self.dropouts) > i:
                x = self.dropouts[i](x)
        logit = self.fc(x)  # (batch_size, num_heads, 1)
        return logit


class LinearCrossNetworkV3(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads=1, layer_norm=True, batch_norm=False, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.ws = nn.ModuleList()
        self.bs = nn.ParameterList()
        for i in range(num_layers):
            self.ws.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.bs.append(nn.Parameter(torch.zeros(input_dim)))
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(input_dim // 2))
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))
        self.masker = nn.ReLU()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            H = self.ws[i](x)
            if len(self.batch_norms) > i:
                H = self.batch_norms[i](H)
            if len(self.layer_norms) > i:
                H_norm = self.layer_norms[i](H)
                mask = self.masker(H_norm)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x0 * (H + self.bs[i]) + x
            if len(self.dropouts) > i:
                x = self.dropouts[i](x)
        logit = self.fc(x)
        return logit


class DeepCrossNetworkV3(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.num_heads = args.num_heads  # Add num_heads in your args
        self.embedding = MultiHeadFeaturesEmbedding(self.field_dims, args.embed_dim, self.num_heads)
        self.input_dim = len(self.field_dims) * args.embed_dim
        self.exponential_cross_network = ExponentialCrossNetworkV3(
            input_dim=self.input_dim,
            num_layers=args.num_deep_cross_layers,
            num_heads=self.num_heads,
            layer_norm=args.layer_norm,
            batch_norm=args.batch_norm,
            dropout=args.deep_net_dropout
        )
        self.linear_cross_network = LinearCrossNetworkV3(
            input_dim=self.input_dim,
            num_layers=args.num_shallow_cross_layers,
            num_heads=self.num_heads,
            layer_norm=args.layer_norm,
            batch_norm=args.batch_norm,
            dropout=args.shallow_net_dropout
        )
        if args.task == 'classification':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        # x: (batch_size, num_fields)
        multihead_feature_emb = self.embedding(x)  # shape: (batch_size, num_heads, input_dim)
        # ExponentialCrossNetworkV3
        dlogit = self.exponential_cross_network(multihead_feature_emb).mean(dim=1)  # shape: (batch_size, 1)
        # LinearCrossNetworkV3
        slogit = self.linear_cross_network(multihead_feature_emb).mean(dim=1)
        # Combine logits
        logit = 0.5 * (dlogit + slogit)
        output = self.output_activation(logit.squeeze(1))
        return output

class Image_DeepCrossNetworkV3(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.num_heads = args.num_heads
        self.embed_dim = args.embed_dim

        # 범주형 특징 임베딩
        self.embedding = MultiHeadFeaturesEmbedding(self.field_dims, self.embed_dim, self.num_heads)

        # 이미지 특징을 위한 CNN
        self.cnn = CNN_Base(
            input_size=(3, args.img_size, args.img_size),
            channel_list=args.channel_list,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            batchnorm=args.cnn_batchnorm,
            dropout=args.cnn_dropout
        )

        # CNN 출력 차원 계산
        cnn_output_dim = np.prod(self.cnn.output_dim)

        # CNN 출력을 임베딩 차원으로 변환
        self.cnn_embedding = nn.Linear(cnn_output_dim, self.embed_dim * self.num_heads)

        # 입력 차원 업데이트 (범주형 임베딩 + 이미지 임베딩)
        self.input_dim = (len(self.field_dims) * self.embed_dim) + self.embed_dim  # 수정된 부분

        # Exponential Cross Network V3
        self.exponential_cross_network = ExponentialCrossNetworkV3(
            input_dim=self.input_dim,
            num_layers=args.num_deep_cross_layers,
            num_heads=self.num_heads,
            layer_norm=args.layer_norm,
            batch_norm=args.batch_norm,
            dropout=args.deep_net_dropout
        )

        # Linear Cross Network V3
        self.linear_cross_network = LinearCrossNetworkV3(
            input_dim=self.input_dim,
            num_layers=args.num_shallow_cross_layers,
            num_heads=self.num_heads,
            layer_norm=args.layer_norm,
            batch_norm=args.batch_norm,
            dropout=args.shallow_net_dropout
        )

        if args.task == 'classification':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        # x: [user_book_vector, img_vector]
        user_book_vector, img_vector = x[0], x[1]

        # 범주형 특징 임베딩
        multihead_feature_emb = self.embedding(user_book_vector)  # (batch_size, num_heads, input_dim_categorical)

        # 이미지 특징 임베딩
        img_feature = self.cnn(img_vector)  # (batch_size, out_channels, H, W)
        img_feature = img_feature.view(-1, np.prod(self.cnn.output_dim))  # (batch_size, cnn_output_dim)
        img_feature_emb = self.cnn_embedding(img_feature)  # (batch_size, embed_dim * num_heads)
        img_feature_emb = img_feature_emb.view(-1, self.num_heads, self.embed_dim)  # (batch_size, num_heads, embed_dim)
        
        # 범주형 임베딩과 이미지 임베딩 결합
        combined_feature = torch.cat([multihead_feature_emb, img_feature_emb], dim=2)  # (batch_size, num_heads, total_input_dim)

        # Exponential Cross Network V3
        dlogit = self.exponential_cross_network(combined_feature).mean(dim=1)  # (batch_size, 1)

        # Linear Cross Network V3
        slogit = self.linear_cross_network(combined_feature).mean(dim=1)  # (batch_size, 1)

        # 로짓 결합
        logit = 0.5 * (dlogit + slogit)
        output = self.output_activation(logit.squeeze(1))
        return output