# =============================================================================
# [Notebook] 01_B_Validation_LagLlama.ipynb
# 주제: [검증 1 보완] SOTA Foundation Model (Lag-Llama) 비교 검증
# 내용: "체리 피킹" 의혹 해소를 위한 Rolling Window (시계열 전진 분석) 적용
# =============================================================================

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import urllib.request

# 경고 억제
warnings.simplefilter("ignore")

# GluonTS Imports
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions

# Lag-Llama Import
try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
except ImportError:
    print("[Error] Lag-Llama 라이브러리가 없습니다.")
    sys.exit(1)

# 상위 폴더 모듈 임포트
sys.path.append(os.path.abspath('..'))
# (사용자 환경에 맞게 경로가 설정되어 있다고 가정)
try:
    from src.data_gen import dual_regime, prep_data
    from src.utils import quantile_loss_stats
except ImportError:
    # src가 없는 경우를 대비한 더미 함수 (실행 시엔 무시됨)
    pass

# =============================================================================
# 1. 데이터 준비
# =============================================================================
print("[1/4] 데이터 생성 및 전처리...")

# HAQR과 동일한 데이터 생성
# (src 모듈이 정상적으로 import 되었다고 가정)
data = dual_regime(total_steps=5000, prob_switch=0.20, stdev=0.0145)
model_data, raw_data = prep_data(data=data, with_flags=True)

# [중요] 인덱스 명시적 변환 (Timezone 제거 및 통일)
raw_data.index = pd.to_datetime(raw_data.index).tz_localize(None)
model_data.index = pd.to_datetime(model_data.index).tz_localize(None)

# float32 변환
raw_data_f32 = raw_data.astype('float32')

# Train/Test Split Point 계산
split_idx = int(len(raw_data) * 0.6)
print(f"  - 전체 데이터: {len(raw_data)}개")
print(f"  - 테스트 시작 인덱스: {split_idx} (날짜: {raw_data.index[split_idx].date()})")

# =============================================================================
# 2. Lag-Llama 모델 로드
# =============================================================================
print("[2/4] Lag-Llama 모델 로드...")

prediction_length = 1
context_length = 32

# 모델 가중치 파일 다운로드 확인
ckpt_path = "lag-llama.ckpt"
if not os.path.exists(ckpt_path):
    print("  - Lag-Llama 가중치 다운로드 중...")
    url = "https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt?download=true"
    urllib.request.urlretrieve(url, ckpt_path)
    print("  - 다운로드 완료.")

# Estimator 설정 (사용자 설정 유지: 8 Layers)
estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=prediction_length,
    context_length=context_length,
    batch_size=64,
    n_layer=8,          
    n_embd_per_head=18, 
    n_head=8,           
    time_feat=True, 
    trainer_kwargs={"accelerator": "auto", "max_epochs": 50} 
)

# Predictor 생성
lightning_module = estimator.create_lightning_module()
transformation = estimator.create_transformation()
predictor = estimator.create_predictor(transformation, lightning_module)
print("  - Predictor 생성 완료.")

# =============================================================================
# 3. Rolling Window 예측 (수정됨)
# =============================================================================
print("[3/4] Rolling Window 예측 수행 (시간이 소요됩니다)...")

rolling_datasets = [] # 여기에는 dict 형태가 들어가야 함

step = 1
test_indices = range(split_idx, len(raw_data_f32), step)

print(f"  - 생성할 예측 샘플 수: {len(test_indices)}개")

for idx in tqdm(test_indices, desc="Preparing Datasets"):
    slice_series = raw_data_f32.iloc[:idx]
    
    # 1. PandasDataset으로 변환 (여기까지는 동일)
    # freq='D'를 명시하면 빈 날짜가 NaN으로 채워질 수 있으니, 
    # 데이터가 평일만 있다면 freq='B'가 좋으나, 안전하게는 freq를 생략하거나 Lag-Llama가 처리하게 둠
    ds_obj = PandasDataset(slice_series, target="rets")
    
    # 2. [핵심 수정] 객체 자체가 아니라, 내부의 데이터(Dict)를 추출
    # PandasDataset은 iterable이므로 next(iter())로 첫 번째(유일한) 시계열 데이터를 꺼냄
    entry = next(iter(ds_obj))
    
    rolling_datasets.append(entry)

print("  - Inference 시작...")

# 이제 rolling_datasets는 [{'start':..., 'target':...}, {...}] 형태의 리스트입니다.
forecast_it, ts_it = make_evaluation_predictions(
    dataset=rolling_datasets, 
    predictor=predictor,
    num_samples=100 
)

forecasts = list(forecast_it)
# tss = list(ts_it) # 메모리 절약을 위해 필요 없다면 주석 처리

# =============================================================================
# 4. 성능 평가 및 저장
# =============================================================================
print("[4/4] 결과 매칭 및 저장...")

lag_llama_preds = []

for f in forecasts:
    # 1. 날짜 추출 및 정규화
    try:
        ts = f.start_date.to_timestamp()
    except:
        ts = f.start_date
    
    date = pd.to_datetime(ts).tz_localize(None).normalize()
    
    # 2. Quantile 추출
    q05 = float(f.quantile(0.05))
    q50 = float(f.quantile(0.50))
    q95 = float(f.quantile(0.95))
    
    lag_llama_preds.append({
        'index': date,
        'q05': q05,
        'q50': q50,
        'q95': q95
    })

# 결과 DataFrame 생성
df_lag = pd.DataFrame(lag_llama_preds)
df_lag = df_lag.drop_duplicates(subset=['index']).set_index('index').sort_index()

# 검증할 정답지(Target) 준비
# raw_data 전체에서 해당 날짜들의 실제 수익률을 가져옴
target_dates = df_lag.index
actuals = raw_data_f32.loc[raw_data_f32.index.isin(target_dates)]['rets']

# 교집합 확인
common_index = df_lag.index.intersection(actuals.index)
df_aligned = df_lag.loc[common_index]
y_true_aligned = actuals.loc[common_index].values

if len(df_aligned) == 0:
    print("\n[Critical Error] 여전히 날짜 매칭에 실패했습니다.")
    print("Pred Dates Example:", df_lag.index[:3])
    print("Real Dates Example:", actuals.index[:3])
else:
    print(f"  - 매칭 성공! 평가 샘플 수: {len(df_aligned)}")

    # Pinball Loss 계산
    # (M1 Signal 필터링 등은 src.utils.quantile_loss_stats 내부 로직에 맡기거나 여기서 필터링)
    # 여기서는 전체 Test 기간에 대해 계산 후 저장
    pred_quantiles = df_aligned[['q05', 'q50', 'q95']].values
    metrics = quantile_loss_stats(y_true_aligned, pred_quantiles, prefix='lag_llama')

    print("\n[Lag-Llama (Rolling) 성능]")
    print(metrics)

    # 결과 CSV 저장
    save_path = '../results/experiment_01_B_lag_llama.csv'
    metrics.to_csv(save_path, index=False)
    
    # 원본 예측 데이터도 저장 (나중에 분석용)
    df_aligned['actual'] = y_true_aligned
    df_aligned.to_csv('../results/experiment_01_B_lag_llama_predictions.csv')
    
    print(f"[INFO] 요약 결과 저장: {save_path}")
    print(f"[INFO] 상세 예측 저장: ../results/experiment_01_B_lag_llama_predictions.csv")

print("\n" + "="*60)
print(" [검증 1 보완] HAQR vs SOTA (Lag-Llama) Rolling 검증 완료")
print("="*60)
