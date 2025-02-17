# Nabi

히브리어 נָבִיא (나비; 예언자라는 뜻)

## Team Members

- 구나영
- 전재영
- 권순준
- 최윤영
- 정한영

## Project Description

### Data

```
data
├── train.csv
├── test.csv
├── X_train.feat_v0.npy
├── X_test.feat_v0.npy
├── y_train.feat_v0.npy
├── infomax300.train.npy
└── infomax300.test.npy
```

- `train.csv`와 `test.csv`는 대회에서 제공한 데이터 파일
- `*.feat_v0.npy`는 `features_v0.ipynb`를 실행시켜서 만든 데이터 파일
- `infomax300.*.npy`는 `make_infomax_dataset.ipynb`를 실행시켜서 만든 데이터 파일

### Submissions

```
submitted
├── final_submisstion.csv
└── submisstion_model.pth
```

- 최종 예측 결과 `.csv` 파일
- 학습된 모델 가중치 `.pth` 파일

### Code

- `submit_model_test.ipynb`: 제출한 모델 가중치를 사용하여 테스트 세트에 대한 예측을 수행
- `train_predict.ipynb`: 모델 학습 및 최종 test set 예측 수행

### MLP_log_ic50_fold_trial5_submit.ipynb code output

- `MLP_Stratified_KFold_log_IC50_YYYY-MM-DD-HH-MM-SS`
  - 학습 로그 생성됨
  - `YYYY-MM-DD-HH-MM-SS`: timestamp
- `submission.csv`: 최종 test set 예측 파일

## Development Environment

### Hardware

- OS: Windows 11
- CPU: Intel(R) Core(TM) i5-1035G1
- GPU: NVIDIA GeForce MX350

### Software

- Conda: 23.7.4
- CUDA Version: 11.8
- NVIDIA-Driver: 522.06
- cuDNN: 8.7.0
- Python: 3.12.4

## Dependencies

- pandas==2.2.2
- mlflow==2.16.0
- matplotlib==3.9.2
- torch==2.4.1+cu118
- optuna==4.0.0
- numpy==1.26.4
- scikit-learn==1.5.1
- jupyterlab==4.2.5

## Installation and Setup

### Step 1. Create a New Conda Environment

```bash
conda create --name Nabi python==3.12 pip
```

### Step 2. Activate the Environment

```bash
conda activate Nabi
```

### Step 3. Install the Required Packages

```bash
pip install -r requirements.txt
```

### Step 4. Install PyTorch with CUDA Support

#### Windows

```bash
conda install pytorch=2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### macOS M2

```bash
conda install pytorch::pytorch=2.3.0 torchvision torchaudio -c pytorch
```

### Step 5. Install Deep Graph Library (DGL)

This is needed for infomax:

```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### Step 6. Start the MLflow Server

```bash
mlflow ui
```

### Step 7. Open and Run the Jupyter Notebooks

- `train_predict.ipynb`: 모델 학습 및 평가, test set 예측 수행
- `submit_model_test.ipynb`: private score로 제출한 학습된 model weights를 통해 test set 예측 수행
