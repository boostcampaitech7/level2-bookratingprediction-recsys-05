import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from .basic_data import basic_data_split
from .context_data import process_context_data


class Text_Dataset(Dataset):
    def __init__(self, user_book_vector, user_summary_vector, book_summary_vector, rating=None, weights=None):
        """
        Parameters
        ----------
        user_book_vector : np.ndarray
            벡터화된 유저와 책 데이터를 입렵합니다.
        user_summary_vector : np.ndarray
            벡터화된 유저에 대한 요약 정보 데이터를 입력합니다.
        book_summary_vector : np.ndarray
            벡터화된 책에 대한 요약 정보 데이터 입력합니다.
        label : np.ndarray
            정답 데이터를 입력합니다.
        ----------
        """
        self.user_book_vector = user_book_vector
        self.user_summary_vector = user_summary_vector
        self.book_summary_vector = book_summary_vector
        self.rating = rating
        self.weights = weights
        
    def __len__(self):
        return self.user_book_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_book_vector' : torch.tensor(self.user_book_vector[i], dtype=torch.long),
                'user_summary_vector' : torch.tensor(self.user_summary_vector[i], dtype=torch.float32),
                'book_summary_vector' : torch.tensor(self.book_summary_vector[i], dtype=torch.float32),
                'rating' : torch.tensor(self.rating[i], dtype=torch.float32),
                'weights': torch.tensor(self.weights[i], dtype=torch.float32)
                } if self.rating is not None else \
                {
                'user_book_vector' : torch.tensor(self.user_book_vector[i], dtype=torch.long),
                'user_summary_vector' : torch.tensor(self.user_summary_vector[i], dtype=torch.float32),
                'book_summary_vector' : torch.tensor(self.book_summary_vector[i], dtype=torch.float32),
                }


def text2_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    args.model_args[args.model].pretrained_model : str
        사전학습된 모델을 설정할 수 있는 parser
    args.model_args[args.model].vector_create : bool
        텍스트 데이터 벡터화 및 저장 여부를 설정할 수 있는 parser
        False로 설정하면 기존에 저장된 벡터를 불러옵니다.

    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """
    train_df = pd.read_csv(args.dataset.data_path + 'processed_train_text.csv')
    test_df = pd.read_csv(args.dataset.data_path + 'processed_test_text.csv')
    

    data = {
            'train':train_df,
            'test':test_df
            }
    
    return data


def text2_data_split(args, data):
    """학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다."""
    data = basic_data_split(args, data)
    
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


def text2_data_loader(args, data):
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
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용
    data : dict
        text_data_load()에서 반환된 데이터

    Returns
    -------
    data : dict
        Text_Dataset 형태의 학습/검증/테스트 데이터를 DataLoader로 변환하여 추가한 후 반환합니다.
    """
    
    # 레이팅에 해당하는 가중치를 매핑
    y_train_weights = data['y_train'].map(data['rating_weights']).values
    if args.dataset.valid_ratio != 0:
        y_valid_weights = data['y_valid'].map(data['rating_weights']).values
    else:
        y_valid_weights = None
    train_dataset = Text_Dataset(
                                data['X_train'][data['field_names']].values,
                                data['X_train']['user_summary_merge_vector'].values,
                                data['X_train']['book_summary_vector'].values,
                                data['y_train'].values,
                                y_train_weights 
                                )
    valid_dataset = Text_Dataset(
                                data['X_valid'][data['field_names']].values,
                                data['X_valid']['user_summary_merge_vector'].values,
                                data['X_valid']['book_summary_vector'].values,
                                data['y_valid'].values,
                                y_train_weights  
                                ) if args.dataset.valid_ratio != 0 else None
    test_dataset = Text_Dataset(
                                data['test'][data['field_names']].values,
                                data['test']['user_summary_merge_vector'].values,
                                data['test']['book_summary_vector'].values,
                                )
    full_train_dataset  = Text_Dataset(
                                      data['X_train_full'][data['field_names']].values,
                                      data['X_train_full']['user_summary_merge_vector'].values,
                                      data['X_train_full']['book_summary_vector'].values,
                                      data['y_train_full'].values,
                                      pd.concat([data['y_train'], data['y_valid']], axis=0).map(data['rating_weights']).values  # 전체 학습 데이터 가중치
                                      )

    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)
    full_train_dataloader = DataLoader(full_train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    data['full_train_dataloader'] = full_train_dataloader  # 추가
    
    return data
