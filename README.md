![header](https://capsule-render.vercel.app/api?type=waving&color=0:EDDFE0,100:B7B7B7&width=max&height=250&section=header&text=Book&nbsp;Rating&nbsp;Prediction&desc=RecSys05-오곡밥&fontSize=40&fontColor=4A4947&&fontAlignY=40)

## 🍚 팀원 소개

|문원찬|안규리|오소영|오준혁|윤건욱|황진욱|
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/a29cbbd9-0cde-495a-bd7e-90f20759f3d1" width="100"/> | <img src="https://github.com/user-attachments/assets/c619ed82-03f3-4d48-9bba-dd60408879f9" width="100"/> | <img src="https://github.com/user-attachments/assets/1b0e54e6-57dc-4c19-97f5-69b7e6f3a9b4" width="100"/> | <img src="https://github.com/user-attachments/assets/67d19373-8cac-4676-bde1-b0637921cf7f" width="100"/> | <img src="https://github.com/user-attachments/assets/f91dd46e-9f1a-42e7-a939-db13692f4098" width="100"/> | <img src="https://github.com/user-attachments/assets/69bbb039-752e-4448-bcaa-b8a65015b778" width="100"/> |
| [![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/WonchanMoon)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/notmandarin)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/irrso)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/ojunhyuk99)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/YoonGeonWook)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/hw01931)|

</br>

## 💡 프로젝트 개요

### 프로젝트 소개
사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 평점을 예측하는 프로젝트입니다.

### 데이터 소개
데이터셋은 사용자 아이디, 위치, 나이, isbn, 책이름, 저자, 출판년도, 출판사, 표지, 언어, 카테고리, 줄거리, 평점 등 다양한 특성으로 구성되어 있습니다.

### 데이터셋 구성
- **train_ratings.csv**: 각 사용자가 책에 대해 평점을 매긴 내역
- **users.csv** : 사용자에 대한 정보
- **books.csv** : 책에 대한 정보
- **Image/** : 책 이미지

### 평가 방식
- **평가 지표**: Root Mean Square Error (RMSE)를 사용하여 예측 성능을 평가합니다.

### 프로젝트 목표
소비자들의 책 구매 결정에 대한 도움을 주기 위해 개인화된 상품 추천을 진행하고자 합니다.

</br>

## 📂폴더구조
```

```
</br>

## ⚙️ 개발 환경
#### OS: Linux (5.4.0-99-generic, x86_64)
#### GPU: Tesla V100-SXM2-32GB (CUDA Version: 12.2)
#### CPU : Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 8 Cores
</br>

## 🔧 기술 스택

#### 프로그래밍 언어 <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=python&logoColor=white"/>

#### 데이터 분석 및 전처리 <img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=numpy&logoColor=white"/>

#### 모델 학습 및 평가 <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat-square&logo=scikitlearn&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-D00000.svg?style=flat-square&logo=keras&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/CatBoost-FECC00.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/DCN-569A31.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/DCN_v2-477B27.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/DCN_v3-335C1D.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/DeepFM-134881.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/FFM-00B388.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/FM-42B883.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/NCF-3B66BC.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/WDN-34567C.svg?style=flat-square&logoColor=white"/> 
  
#### 시각화 도구 <img src="https://img.shields.io/badge/Matplotlib-3F4F75.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/seaborn-221E68.svg?style=flat-square&logoColor=white"/>

#### 개발 환경 <img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=jupyter&logoColor=white"/>

#### 실험 관리 및 추적 <img src="https://img.shields.io/badge/Weights&Biases-FFBE00.svg?style=flat-square&logo=weightsandbiases&logoColor=black"/>
