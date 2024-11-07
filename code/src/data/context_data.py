import numpy as np
import pandas as pd
import regex
import torch
from torch.utils.data import TensorDataset, DataLoader
from .basic_data import basic_data_split

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[2:-2].split(', ')


def split_location(x: str) -> list:
    '''
    Parameters
    ----------
    x : str
        location 데이터

    Returns
    -------
    res : list
        location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
        순서는 country, state, city, ... 입니다.
    '''
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  # remove special characters
    res = [i if i not in ['n/a', ''] else np.nan for i in res]  # change 'n/a' into NaN
    res.reverse()  # reverse the list to get country, state, city, ... order

    return res


def process_context_data(users, books):
    users_ = users.copy()
    books_ = books.copy()

    # category: Fiction/NonFiction/unknown
    books_['category'] = books_['category'].apply(
        lambda x: 'Fiction' if 'fiction' in str(x).lower() else ('NonFiction' if pd.notna(x) else np.nan)
    )
    
    # language: V1 = ['en', 'non_en', 'unknown']
    books_['language'] = books_['language'].apply(lambda x: 'en' if x == 'en' else ('NaN' if pd.isna(x) else 'non_en'))
    # books_['language'] = books_['language'].apply(lambda x: x if x in ['en', 'de', 'es', 'fr', 'it'] else ('NaN' if pd.isna(x) else 'others'))
    
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)  # 1990년대, 2000년대, 2010년대, ...

    # book_author: high_cardinality 처리(상위 120개)
    top_authors = books_['book_author'].value_counts().nlargest(120).index
    books_['book_author'] = books_['book_author'].apply(lambda x: x if x in top_authors else 'Other')
    
    # age_range: 결측치 -1로 대체
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10).fillna(-1).astype(int) # 10대, 20대, 30대, ...
    
    # year_of_publication: 1900년도 미만 제거
    books_ = books_[books_['year_of_publication'] >= 1900].reset_index(drop=True)
    
    # location -> country, city, state -> 상위 50개, 120개, 120개
    users_['location_list'] = users_['location'].apply(lambda x: split_location(x)) 
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    for idx, row in users_.iterrows():
        if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
            fill_country = users_[users_['location_state'] == row['location_state']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_.loc[idx, 'location_country'] = fill_country
        elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
            if not pd.isna(row['location_country']):
                fill_state = users_[(users_['location_country'] == row['location_country']) 
                                    & (users_['location_city'] == row['location_city'])]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                users_.loc[idx, 'location_state'] = fill_state
            else:
                fill_state = users_[users_['location_city'] == row['location_city']]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
                fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
                users_.loc[idx, 'location_country'] = fill_country
                users_.loc[idx, 'location_state'] = fill_state
    
    top_country = users_['location_country'].value_counts().nlargest(50).index
    users_['location_country'] = users_['location_country'].apply(lambda x: x if x in top_country else 'Other')
    top_state = users_['location_state'].value_counts().nlargest(120).index
    users_['location_state'] = users_['location_state'].apply(lambda x: x if x in top_state else 'Other')
    top_city = users_['location_city'].value_counts().nlargest(120).index
    users_['location_city'] = users_['location_city'].apply(lambda x: x if x in top_city else 'Other')
    
    users_ = users_.drop(['location'], axis=1)

    return users_, books_


def context_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    users_, books_ = process_context_data(users, books)
    
    
    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    # 베이스라인에서는 가능한 모든 컬럼을 사용하도록 구성하였습니다.
    # NCF를 사용할 경우, idx 0, 1은 각각 user_id, isbn이어야 합니다.
    user_features = ['user_id', 'age_range', 'location_country', 'location_state', 'location_city']
    book_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category', 'publication_range']
    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'}) if args.model == 'NCF' \
                   else user_features + book_features

    # 선택한 컬럼만 추출하여 데이터 조인
    train_df = train.merge(users_, on='user_id', how='left')\
                    .merge(books_, on='isbn', how='left')[sparse_cols + ['rating']]
    test_df = test.merge(users_, on='user_id', how='left')\
                  .merge(books_, on='isbn', how='left')[sparse_cols]
    all_df = pd.concat([train_df, test_df], axis=0)

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
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
    
    field_dims = [len(label2idx[col]) for col in train_df.columns if col != 'rating']

    # 'num_users'와 'num_items' 계산 및 추가
    num_users = len(label2idx['user_id'])
    num_items = len(label2idx['isbn'])

    data = {
        'train': train_df,
        'test': test_df,
        'field_names': sparse_cols,
        'field_dims': field_dims,
        'label2idx': label2idx,
        'idx2label': idx2label,
        'sub': sub,
        'num_users': num_users,  # 추가
        'num_items': num_items   # 추가
    }

    return data


def context_data_split(args, data):
    '''data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.'''
    data = basic_data_split(args, data)
    
    # 전체 학습 데이터를 위한 X_train_full, y_train_full 생성
    data['X_train_full'] = pd.concat([data['X_train'], data['X_valid']], axis=0).reset_index(drop=True)
    data['y_train_full'] = pd.concat([data['y_train'], data['y_valid']], axis=0).reset_index(drop=True)
    
    return data


def context_data_loader(args, data):
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
        context_data_load 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.dataset.valid_ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    # 전체 학습 데이터를 위한 데이터셋 생성
    full_train_dataset = TensorDataset(torch.LongTensor(data['X_train_full'].values), torch.LongTensor(data['y_train_full'].values))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)
    # 전체 학습 데이터를 위한 데이터로더 생성
    full_train_dataloader = DataLoader(full_train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    data['full_train_dataloader'] = full_train_dataloader  # 추가
    
    return data
