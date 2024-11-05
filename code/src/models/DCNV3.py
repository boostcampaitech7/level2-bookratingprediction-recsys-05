import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import LayerNorm
from ._helpers import FeaturesEmbedding, MLP_Base


# Embedding & Reshape Layer
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_sizes, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes])
        self.embed_dim = embed_dim

    def forward(self, *inputs):
        embeddings = [self.embeddings[i](inputs[i]) for i in range(len(inputs))]
        # Chunk operation to split embeddings into two views
        e_a, e_b = zip(*[self.chunk(embed) for embed in embeddings])
        e_a = torch.cat(e_a, dim=-1)  # Concatenate all a embeddings
        e_b = torch.cat(e_b, dim=-1)  # Concatenate all b embeddings
        return e_a, e_b
    
    def chunk(self, embed):
        # Assuming embed is of shape [batch_size, field_size, embed_dim]
        # Chunk into two views: e_a and e_b
        return torch.chunk(embed, 2, dim=-1)
    
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, mask_rate=0.5):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.mask_rate = mask_rate
        self.weights = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.biases = nn.ModuleList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x):
        for l in range(self.num_layers):
            cross = self.weights[l](x) + self.biases[l]
            cross = cross * self.self_mask(cross)
            x = x + cross
        return x

    def self_mask(self, x):
        # Self-Mask operation: apply LayerNorm and element-wise multiplication
        layer_norm = F.layer_norm(x, x.size())
        mask = torch.bernoulli(torch.sigmoid(layer_norm))  # Bernoulli mask
        return mask
    
class FusionLayer(nn.Module):
    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self.fc_d = nn.Linear(input_dim, 1)
        self.fc_s = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_d, x_s):
        y_d = self.sigmoid(self.fc_d(x_d))  # Prediction from DCNv3
        y_s = self.sigmoid(self.fc_s(x_s))  # Prediction from SCNv3
        y = (y_d + y_s) / 2  # Combine predictions
        return y
    
class TriBCELoss(nn.Module):
    def __init__(self):
        super(TriBCELoss, self).__init__()

    def forward(self, y_true, y_pred_d, y_pred_s, loss_d, loss_s):
        # Binary Cross-Entropy loss for DCNv3, SCNv3
        bce_loss = F.binary_cross_entropy(y_pred_d, y_true) + F.binary_cross_entropy(y_pred_s, y_true)
        # Compute Tri-BCE loss
        weight_d = torch.maximum(torch.tensor(0.0), loss_d - bce_loss)
        weight_s = torch.maximum(torch.tensor(0.0), loss_s - bce_loss)
        tri_loss = bce_loss + weight_d * loss_d + weight_s * loss_s
        return tri_loss
    
class SDCNv3(nn.Module):
    def __init__(self, vocab_sizes, embed_dim, num_cross_layers, input_dim):
        super(SDCNv3, self).__init__()
        self.embedding_layer = EmbeddingLayer(vocab_sizes, embed_dim)
        self.cross_network_d = CrossNetwork(input_dim, num_cross_layers)  # DCNv3
        self.cross_network_s = CrossNetwork(input_dim, num_cross_layers)  # SCNv3
        self.fusion_layer = FusionLayer(input_dim)
        self.loss_fn = TriBCELoss()

    def forward(self, x_p, x_a, x_c, y_true):
        e_a, e_b = self.embedding_layer(x_p, x_a, x_c)  # Embedding & reshape
        x = torch.cat([e_a, e_b], dim=-1)  # Combine embeddings

        # Forward pass through cross networks
        x_d = self.cross_network_d(x)
        x_s = self.cross_network_s(x)

        # Fusion and final prediction
        y_pred_d = self.fusion_layer(x_d, x_s)
        y_pred_s = self.fusion_layer(x_s, x_d)

        # Calculate loss
        loss_d = F.binary_cross_entropy(y_pred_d, y_true)
        loss_s = F.binary_cross_entropy(y_pred_s, y_true)
        loss = self.loss_fn(y_true, y_pred_d, y_pred_s, loss_d, loss_s)
        return y_pred_d, loss