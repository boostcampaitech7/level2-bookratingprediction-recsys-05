import time
import argparse
import pandas as pd
import numpy as np
from types import SimpleNamespace

import os
import sys
# 스크립트 파일의 절대 경로를 기준으로 상위 디렉토리 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))  # Adjust based on actual structure
sys.path.append(project_root)

from src.data import context_data_load, context_data_split, context_data_loader
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor

import optuna
import joblib
import json

# Argument Parser 설정
parser = argparse.ArgumentParser(description='CatBoost with Optuna Hyperparameter Optimization')

# 최상위 인자
parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
parser.add_argument('--save_time', type=str, default=None, help='Timestamp for saving models and logs')
parser.add_argument('--model', type=str, default='CatBoost', help='Model Name')

# Dataset 관련 인자 그룹
dataset_group = parser.add_argument_group('dataset')
dataset_group.add_argument('--data_path', type=str, default='data/', help='Path to the data directory')
dataset_group.add_argument('--valid_ratio', type=float, default=0.2, help='Validation set ratio')
dataset_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

# WandB 관련 인자 그룹
wandb_group = parser.add_argument_group('wandb')
wandb_group.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
wandb_group.add_argument('--wandb_project', type=str, default='CatBoost_Optuna_Project', help='WandB project name')
wandb_group.add_argument('--run_name', type=str, default=None, help='WandB run name')

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
    save_time = now_date + '_' + now_hour
else:
    save_time = args.save_time
    
# WandB 초기화
if args.wandb:
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.run_name if args.run_name else f"Run_{save_time}",
        config={
            'trials': args.trials,
            'data_path': args.dataset.data_path,
            'valid_ratio': args.dataset.valid_ratio,
            'seed': args.dataset.seed
        }
    )    


# 데이터 로드
data = context_data_load(args) # IMAGE X

# 학습 데이터와 검증 데이터 분할
feature = data['train'].drop(columns='rating')
label = data['train']['rating']
x_test = data['test']

X_train, X_val, y_train, y_val = train_test_split(feature, label, test_size = 0.2, shuffle = True,random_state = 42)

## optuna
def objective(trial):
    param = {
        # 1) 하이퍼 파리미터 목록
        'task_type': 'GPU',
        'devices': 'cuda',
        'iterations':trial.suggest_int("iterations", 500, 3000),
        'od_wait':trial.suggest_int('od_wait', 500, 1500),
        'learning_rate' : trial.suggest_float('learning_rate',0.001, 1),
        'random_strength': trial.suggest_float('random_strength',0,30),
        'depth': trial.suggest_int('depth',3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'bagging_temperature' :trial.suggest_float('bagging_temperature', 0, 5),
        "l2_leaf_reg": trial.suggest_float('l2_leaf_reg', 1, 20),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }

    try:
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, cat_features=list(feature.columns), verbose=True, eval_set=[(X_val, y_val)], early_stopping_rounds=100)
        pred = model.predict(X_val)
        RMSE = np.sqrt(mean_squared_error(y_val, pred))
    except Exception as e:
        print(f"오류 발생: {e}")
        return float('inf')  # Optuna에게 최악의 결과를 전달
    return RMSE

# Optuna 스터디 생성 및 최적화 실행
study = optuna.create_study(direction='minimize', study_name='catboost_regressor', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=args.trials,n_jobs=1)

# 최적 하이퍼파라미터 출력
print("최적 하이퍼파라미터:", study.best_params)
print("최적 RMSE:", study.best_value)

## KFold 교차 검증
def perform_kfold_cv(best_params, X, y, categorical_features, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**best_params, task_type='GPU' if args.wandb else 'CPU',
                                 devices='0' if args.wandb else None, verbose=False)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            early_stopping_rounds=100,
            cat_features=categorical_features
        )
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)
        print(f"Fold {fold+1}: RMSE = {rmse:.4f}")

        if args.wandb:
            wandb.log({f'Fold_{fold+1}_RMSE': rmse})

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
    study.best_params,
    X_full_train,
    y_full_train,
    categorical_features=list(feature.columns),
    n_splits=5,
    seed=args.dataset.seed
)

# WandB에 KFold 결과 로깅
if args.wandb:
    wandb.log({
        'KFold_mean_RMSE': mean_rmse,
        'KFold_std_RMSE': std_rmse
    })
    
# WandB 종료
if args.wandb:
    wandb.finish()

# 최적의 하이퍼파라미터로 전체 학습 데이터로 재학습
def retrain_on_full_data(best_params, X, y, categorical_features):
    model = CatBoostRegressor(**best_params, task_type='GPU', devices='0', verbose=False)
    model.fit(
        X, y,
        cat_features=categorical_features
    )
    return model

# 재학습
model_tuned = retrain_on_full_data(
    study.best_params,
    X_full_train,
    y_full_train,
    categorical_features=list(feature.columns)  # 카테고리 특성 지정
)
result = model_tuned.predict(data['test'])

# 모델 저장
os.makedirs('saved_models/', exist_ok=True)
model_filename = f'saved_models/catboost_optuna_{save_time}.pkl'
joblib.dump(model_tuned, model_filename)
print(f"모델이 저장되었습니다: {model_filename}")

# 최적 하이퍼파라미터 저장
os.makedirs(f'log/{save_time}_catboost', exist_ok=True)
hyperparams_filename = f'log/{save_time}_catboost/catboost_best_hyperparameters_{save_time}.json'
with open(hyperparams_filename, 'w') as f:
    json.dump(study.best_params, f)
print(f"최적 하이퍼파라미터가 저장되었습니다: {hyperparams_filename}")


# 제출 파일 생성
os.makedirs('submit/', exist_ok=True)
submission = pd.read_csv(os.path.join(args.data_path, 'sample_submission.csv'))
submission['rating'] = result
submission_filename = f'submit/catboost_optuna_{save_time}.csv'
submission.to_csv(submission_filename, index=False)
print(f"제출 파일이 저장되었습니다: {submission_filename}")