import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def basic_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다
    """

    ######################## DATA LOAD
    # users = pd.read_csv(args.dataset.data_path + 'users.csv')
    # books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train_df = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test_df = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    all_df = pd.concat([train_df, test_df], axis=0)
    
    sparse_cols = ['user_id', 'isbn']

    # 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        train_df[col] = train_df[col].fillna('unknown')
        test_df[col] = test_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].map(label2idx[col])
        test_df[col] = test_df[col].map(label2idx[col])

    field_dims = [len(label2idx[col]) for col in sparse_cols]

    data = {
            'train':train_df,
            'test':test_df.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }


    return data


def basic_data_split(args, data):
    """
    Parameters
    ----------
    args.dataset.valid_ratio : float
        Train/Valid split 비율을 입력합니다.
    args.seed : int
        데이터 셔플 시 사용할 seed 값을 입력합니다.

    Returns
    -------
    data : dict
        data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.
    """
    if args.dataset.valid_ratio == 0:
        data['X_train'] = data['train'].drop('rating', axis=1)
        data['y_train'] = data['train']['rating']

    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            data['train'].drop(['rating'], axis=1),
            data['train']['rating'],
            test_size=args.dataset.valid_ratio,
            random_state=args.seed,
            shuffle=True
        )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    # 레이팅 빈도수 계산
    rating_counts = data['y_train'].value_counts()
    total_counts = len(data['y_train'])
    rating_freq = rating_counts / total_counts
    # 가중치 계산 (빈도의 역수)
    rating_weights = 1 / rating_freq
    # 가중치 정규화 (평균으로 나누어줌)
    rating_weights = rating_weights / rating_weights.mean()
    # 가중치 저장
    data['rating_weights'] = rating_weights
    
    # 전체 학습 데이터를 위한 X_train_full, y_train_full 생성
    data['X_train_full'] = pd.concat([data['X_train'], data['X_valid']], axis=0).reset_index(drop=True)
    data['y_train_full'] = pd.concat([data['y_train'], data['y_valid']], axis=0).reset_index(drop=True)
    
    return data


def basic_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
    data : dict
        basic_data_split 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    # 레이팅에 해당하는 가중치를 매핑
    y_train_weights = data['y_train'].map(data['rating_weights']).values
    if args.dataset.valid_ratio != 0:
        y_valid_weights = data['y_valid'].map(data['rating_weights']).values
    else:
        y_valid_weights = None
    
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values), torch.FloatTensor(y_train_weights))
    if args.dataset.valid_ratio != 0:
        valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values), torch.FloatTensor(y_valid_weights))
    else:
        valid_dataset = None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    # 전체 학습 데이터를 위한 데이터셋 생성
    y_train_full_weights = pd.concat([data['y_train'], data['y_valid']], axis=0).map(data['rating_weights']).values
    full_train_dataset = TensorDataset(torch.LongTensor(data['X_train_full'].values), torch.LongTensor(data['y_train_full'].values), torch.FloatTensor(y_train_full_weights))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)
    # 전체 학습 데이터를 위한 데이터로더 생성
    full_train_dataloader = DataLoader(full_train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    data['full_train_dataloader'] = full_train_dataloader  # 추가
    
    return data
