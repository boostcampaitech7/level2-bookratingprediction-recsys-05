import sys
import os
import time
import argparse
from types import SimpleNamespace

# 현재 파일의 절대 경로를 기반으로 프로젝트 루트 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

# 필요한 모듈 임포트
import pandas as pd
from src.data.context_data import context_data_load
from src.loss.loss import RMSELoss
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
import joblib
import json
import numpy as np

# NCF 모델 클래스 임포트 (NCF 모델이 src/models/에 구현되어 있다고 가정)
from src.models.NCF import NeuralCollaborativeFiltering as NCF  # NCF 모델이 src/models/NCF.py에 정의되어 있어야 합니다.

# Argument Parser 설정
parser = argparse.ArgumentParser(description='NCF with Hyperopt Hyperparameter Optimization')

# 최상위 인자
parser.add_argument('--trials', type=int, default=50, help='Number of Hyperopt trials')
parser.add_argument('--save_time', type=str, default=None, help='Timestamp for saving models and logs')
parser.add_argument('--model', type=str, default='NCF', help='Model Name')

# Dataset 관련 인자 그룹
dataset_group = parser.add_argument_group('dataset')
dataset_group.add_argument('--data_path', type=str, default='data/', help='Path to the data directory')
dataset_group.add_argument('--valid_ratio', type=float, default=0.2, help='Validation set ratio')
dataset_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

args = parser.parse_args()

# `dataset` 네임스페이스 생성
args.dataset = SimpleNamespace(
    data_path=args.data_path,
    valid_ratio=args.valid_ratio,
    seed=args.seed
)

# 저장 시각 설정
if args.save_time is None:
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%H%M%S', now)
    save_time = f"{now_date}_{now_hour}"
else:
    save_time = args.save_time

## DATA
data = context_data_load(args)  # 데이터 로드

# 데이터 딕셔너리 키 확인 (디버깅용)
print("Data keys:", data.keys())

# 학습 데이터와 검증 데이터 분할
feature = data['train'].drop(columns='rating')
label = data['train']['rating']
x_test = data['test']

X_train, X_val, y_train, y_val = train_test_split(
    feature, label, test_size=args.dataset.valid_ratio, shuffle=True, random_state=args.dataset.seed
)

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Hyperopt를 사용한 하이퍼파라미터 최적화
def objective(space):
    try:
        # 하이퍼파라미터 설정
        embed_dim = int(space['embed_dim'])
        mlp_dims = [int(dim) for dim in space['mlp_dims'].split(',')]
        dropout = space['dropout']
        optimizer_name = space['optimizer']
        lr = space['lr']
        weight_decay = space['weight_decay']
        
        # 모델 하이퍼파라미터 네임스페이스 생성
        model_args = SimpleNamespace(
            embed_dim=embed_dim,
            mlp_dims=mlp_dims,
            dropout=dropout,
            batchnorm=True  # 배치 정규화는 고정
        )

        # 모델 초기화
        model = NCF(
            args=model_args,
            data=data
        ).to(device)

        # 손실 함수 및 옵티마이저 설정
        criterion = RMSELoss()
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            # 기본 옵티마이저는 Adam
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 학습 루프
        epochs = 10
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            # 사용자 및 아이템 인덱스를 하나의 텐서로 결합
            inputs = torch.tensor(np.stack([X_train['user_id'].values, X_train['isbn'].values], axis=1), dtype=torch.long).to(device)
            targets = torch.tensor(y_train.values, dtype=torch.float32).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 검증 성능 평가
            model.eval()
            with torch.no_grad():
                val_inputs = torch.tensor(np.stack([X_val['user_id'].values, X_val['isbn'].values], axis=1), dtype=torch.long).to(device)
                val_targets = torch.tensor(y_val.values, dtype=torch.float32).to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_rmse = torch.sqrt(val_loss).item()

        return {'loss': val_rmse, 'status': STATUS_OK}
    except KeyError as e:
        print(f"KeyError in objective function: {e}")
        return {'loss': np.inf, 'status': STATUS_FAIL}

# Hyperopt 검색 공간 정의
space = {
    'embed_dim': hp.quniform('embed_dim', 16, 128, 16),
    'mlp_dims': hp.choice('mlp_dims', ['64,32,16', '128,64,32', '256,128,64']),
    'dropout': hp.uniform('dropout', 0.0, 0.5),
    'optimizer': hp.choice('optimizer', ['Adam']),
    'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-5), np.log(1e-3))
}

# Trials 객체 생성
trials = Trials()

# Hyperopt 최적화 실행
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=args.trials,
    trials=trials,
    rstate=np.random.default_rng(42)
)

# 선택한 하이퍼파라미터 매핑 (hp.choice 사용 시)
# space_choices = {
#     'mlp_dims': ['64,32,16', '128,64,32', '256,128,64'],
#     'optimizer': ['Adam', 'SGD', 'RMSprop']
# }

# best_mlp_dims = space_choices['mlp_dims'][best['mlp_dims']]
# best_optimizer = space_choices['optimizer'][best['optimizer']]

# 최적 하이퍼파라미터 재정의
best_params = {
    'embed_dim': int(best['embed_dim']),
    'mlp_dims': best['mlp_dims'],
    'dropout': best['dropout'],
    'optimizer': best['optimizer'],
    'lr': best['lr'],
    'weight_decay': best['weight_decay']
}

print("Best hyperparameters:", best_params)

## KFold 교차 검증 추가
def perform_kfold_cv(best_params, X, y, categorical_features, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 모델 하이퍼파라미터 네임스페이스 생성
        model_args = SimpleNamespace(
            embed_dim=best_params['embed_dim'],
            mlp_dims=[int(dim) for dim in best_params['mlp_dims'].split(',')],
            dropout=best_params['dropout'],
            batchnorm=True  # 배치 정규화는 고정
        )

        # 모델 초기화
        model = NCF(
            args=model_args,
            data=data
        ).to(device)
        
        # 손실 함수 및 옵티마이저 설정
        criterion = RMSELoss()
        optimizer_name = best_params['optimizer']
        lr = best_params['lr']
        weight_decay = best_params['weight_decay']
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 학습 루프
        epochs = 10
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            # 사용자 및 아이템 인덱스를 하나의 텐서로 결합
            inputs = torch.tensor(np.stack([X_train_fold['user_id'].values, X_train_fold['isbn'].values], axis=1), dtype=torch.long).to(device)
            targets = torch.tensor(y_train_fold.values, dtype=torch.float32).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # 검증 성능 평가
            model.eval()
            with torch.no_grad():
                val_inputs = torch.tensor(np.stack([X_val_fold['user_id'].values, X_val_fold['isbn'].values], axis=1), dtype=torch.long).to(device)
                val_targets = torch.tensor(y_val_fold.values, dtype=torch.float32).to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_rmse = torch.sqrt(val_loss).item()
        
        rmse_scores.append(val_rmse)
        print(f"Fold {fold+1}: RMSE = {val_rmse:.4f}")
            
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    print(f"KFold Mean RMSE: {mean_rmse:.4f}")
    print(f"KFold Std RMSE: {std_rmse:.4f}")
    
    return mean_rmse, std_rmse

# 전체 학습 데이터 준비 (학습 세트 + 검증 세트)
X_full_train = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full_train = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

# KFold 교차 검증 수행
mean_rmse, std_rmse = perform_kfold_cv(
    best_params,
    X_full_train,
    y_full_train,
    categorical_features=list(feature.columns),
    n_splits=5,
    seed=args.dataset.seed
)

## 재학습 후 예측
def retrain_on_full_data(best_params, X, y, categorical_features):
    # 모델 하이퍼파라미터 네임스페이스 생성
    model_args = SimpleNamespace(
        embed_dim=best_params['embed_dim'],
        mlp_dims=[int(dim) for dim in best_params['mlp_dims'].split(',')],
        dropout=best_params['dropout'],
        batchnorm=True  # 배치 정규화는 고정
    )

    # 모델 초기화
    model = NCF(
        args=model_args,
        data=data
    ).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = RMSELoss()
    optimizer_name = best_params['optimizer']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 학습 루프
    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # 사용자 및 아이템 인덱스를 하나의 텐서로 결합
        inputs = torch.tensor(np.stack([X['user_id'].values, X['isbn'].values], axis=1), dtype=torch.long).to(device)
        targets = torch.tensor(y.values, dtype=torch.float32).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 검증 성능 평가 (optional)
        model.eval()
        with torch.no_grad():
            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, targets)
            val_rmse = torch.sqrt(val_loss).item()
        
    return model

# 모델 재학습
model_tuned = retrain_on_full_data(
    best_params,
    X_full_train,
    y_full_train,
    categorical_features=list(feature.columns)
)

# 테스트 데이터 예측
model_tuned.eval()
with torch.no_grad():
    test_users = x_test['user_id'].values
    test_items = x_test['isbn'].values
    # 사용자 및 아이템 인덱스를 하나의 텐서로 결합
    test_inputs = torch.tensor(np.stack([test_users, test_items], axis=1), dtype=torch.long).to(device)
    result = model_tuned(test_inputs).cpu().numpy()

# 모델 저장
os.makedirs('saved_models/', exist_ok=True)
model_filename = f'saved_models/NCF_Hyperopt_{save_time}.pt'
torch.save(model_tuned.state_dict(), model_filename)
print(f"모델이 저장되었습니다: {model_filename}")

# 최적 하이퍼파라미터 저장
os.makedirs(f'log/{save_time}_ncf', exist_ok=True)
hyperparams_filename = f'log/{save_time}_ncf/ncf_best_hyperparameters_{save_time}.json'
with open(hyperparams_filename, 'w') as f:
    json.dump(best_params, f)
print(f"최적 하이퍼파라미터가 저장되었습니다: {hyperparams_filename}")

# 제출 파일 생성
os.makedirs('submit/', exist_ok=True)
submission = pd.read_csv(os.path.join(args.dataset.data_path, 'sample_submission.csv'))
submission['rating'] = result
submission_filename = f'submit/NCF_Hyperopt_{save_time}.csv'
submission.to_csv(submission_filename, index=False)
print(f"제출 파일이 저장되었습니다: {submission_filename}")