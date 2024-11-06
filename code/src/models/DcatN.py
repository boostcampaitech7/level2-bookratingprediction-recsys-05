import torch
import torch.nn as nn
import catboost as cb
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

class DeepCatBoost(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        
        # Deep Part (Neural Network)
        self.cn = CrossNetwork(self.embed_output_dim, args.cross_layer_num)
        self.mlp = MLP_Base(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)
        self.cd_linear = nn.Linear(args.mlp_dims[-1], 1, bias=False)  # Linear output for regression
        
        # Initialize CatBoost model (Wide part)
        self.catboost_model = cb.CatBoostRegressor(iterations=100, loss_function='RMSE', verbose=False)
        
        # Initialize weights for the neural network
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        Train both the CatBoost model (wide part) and the neural network (deep part).
        """
        # Train CatBoost (wide part)
        self.catboost_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)

        # Train Deep learning part
        X_train_embedded = self.embedding(torch.tensor(X_train).long()).view(-1, self.embed_output_dim)
        X_valid_embedded = self.embedding(torch.tensor(X_valid).long()).view(-1, self.embed_output_dim)
        
        # Train the deep model using neural network
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # Using MSELoss to compute RMSE

        for epoch in range(10):  # Example training loop
            self.train()
            optimizer.zero_grad()

            # Forward pass
            output = self(X_train_embedded)
            loss = criterion(output, torch.tensor(y_train, dtype=torch.float32))
            loss.backward()
            optimizer.step()

            # Validation loss
            self.eval()
            with torch.no_grad():
                valid_output = self(X_valid_embedded)
                valid_loss = criterion(valid_output, torch.tensor(y_valid, dtype=torch.float32))
            
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}, Validation Loss (MSE): {valid_loss.item()}")

    def forward(self, x: torch.Tensor):
        # Deep part (neural network)
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        deep_output = self.cd_linear(x_out)

        # Wide part (CatBoost)
        catboost_output = self.catboost_model.predict(x.detach().numpy())

        # Combine the outputs from the deep and wide parts
        combined_output = torch.cat((deep_output, torch.tensor(catboost_output, dtype=torch.float32)), dim=1)

        # Example: Taking the mean of the combined outputs for regression
        final_output = combined_output.mean(dim=1)
        return final_output.squeeze(1)