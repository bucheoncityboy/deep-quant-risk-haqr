# =============================================================================
# [File] 02_Validation_Uncertainty.py (수정본)
# 내용: Uncertainty 검증 (Scale-Up 모델 호환)
# =============================================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 상위 폴더 모듈 임포트
sys.path.append(os.path.abspath('..'))
from src.data_gen import dual_regime, prep_data, denoise_wavelet
from src.models import build_haqr_model
from src.utils import QUANTILES 

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

# 1. 설정
TARGET_ALPHA = 0.1
TARGET_COVERAGE = 0.90

# 2. 데이터 생성 (Raw Data 사용)
data = dual_regime(total_steps=5000, prob_switch=0.20, stdev=0.0145)
model_data, _ = prep_data(data=data, with_flags=True)

train_df, temp_df = train_test_split(model_data, test_size=0.5, shuffle=False)
cal_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=False)

features_info = ['rets', 'rets2', 'rets3']
features_regime = ['rets', 'rets2', 'rets3', 'regime', 'historical_vol']

# Raw Data 사용 (denoise_wavelet은 원본 반환)
X_test_trend = denoise_wavelet(test_df[features_info].values)
X_test_market = test_df[['regime', 'historical_vol']].values

scaler_trend = StandardScaler().fit(train_df[features_info].values)
X_test_trend_sc = scaler_trend.transform(X_test_trend)
scaler_market = StandardScaler().fit(train_df[['regime', 'historical_vol']].values)
X_test_market_sc = scaler_market.transform(X_test_market)

X_train_lgbm = train_df[features_regime]
X_cal_lgbm = cal_df[features_regime]
X_test_lgbm = test_df[features_regime]
y_train = train_df['target_rets']
y_cal = cal_df['target_rets']
y_test = test_df['target_rets']

# 3. HAQR 모델 로드 (Scale-Up 호환 수정)
# [수정] h_dim=32 추가 (저장된 가중치와 크기를 맞춰야 함)
haqr_model, _, _ = build_haqr_model(
    num_trend_features=X_test_trend.shape[1],
    num_market_features=X_test_market.shape[1],
    h_dim=32  # <--- 핵심 수정 사항
)

weights_path = '../weights/haqr_best_model.weights.h5'
if os.path.exists(weights_path):
    haqr_model.load_weights(weights_path)
    print("  - Pre-trained Weights 로드 완료.")
else:
    raise FileNotFoundError("가중치 파일이 없습니다.")

# HAQR 예측
scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
pred_haqr_sc = haqr_model.predict(
    {'trend_input': X_test_trend_sc, 'market_input': X_test_market_sc}, verbose=0
)
pred_haqr = scaler_y.inverse_transform(pred_haqr_sc)
haqr_lower = pred_haqr[:, 0]
haqr_upper = pred_haqr[:, 2]

# 4. SOTA (LGBM + CQR)
lgbm_low = LGBMRegressor(objective='quantile', alpha=0.05, verbose=-1)
lgbm_high = LGBMRegressor(objective='quantile', alpha=0.95, verbose=-1)
lgbm_low.fit(X_train_lgbm, y_train)
lgbm_high.fit(X_train_lgbm, y_train)

cal_low = lgbm_low.predict(X_cal_lgbm)
cal_high = lgbm_high.predict(X_cal_lgbm)
scores = np.maximum(cal_low - y_cal, y_cal - cal_high)
q_hat = np.quantile(scores, np.clip((1 - TARGET_ALPHA) * (1 + 1/len(y_cal)), 0, 1))

test_pred_low = lgbm_low.predict(X_test_lgbm)
test_pred_high = lgbm_high.predict(X_test_lgbm)
cqr_lower = test_pred_low - q_hat
cqr_upper = test_pred_high + q_hat

# 5. 평가 및 시각화
def evaluate_uncertainty(y_true, y_lower, y_upper, model_name):
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    picp = np.mean(covered)
    width = np.mean(y_upper - y_lower)
    return {'Model': model_name, f'PICP (Target {TARGET_COVERAGE})': picp, 'MPIW (Width)': width}

metrics_haqr = evaluate_uncertainty(y_test.values, haqr_lower, haqr_upper, 'HAQR (Intrinsic)')
metrics_cqr = evaluate_uncertainty(y_test.values, cqr_lower, cqr_upper, 'SOTA (LGBM+CQR)')
metrics_raw = evaluate_uncertainty(y_test.values, test_pred_low, test_pred_high, 'LGBM (Raw)')

df_metrics = pd.DataFrame([metrics_haqr, metrics_cqr, metrics_raw])
print("\n[불확실성 성능 비교]")
print(df_metrics.round(4))
df_metrics.to_csv('../results/experiment_02_uncertainty_metrics.csv', index=False)

# 그래프
subset = 150
x_range = range(subset)
plt.figure(figsize=(14, 6))
plt.plot(x_range, y_test.values[:subset], 'k.-', label='Actual Returns', alpha=0.6, linewidth=1)
plt.fill_between(x_range, haqr_lower[:subset], haqr_upper[:subset], color='blue', alpha=0.2, label='HAQR Interval')
plt.plot(x_range, cqr_lower[:subset], 'r--', linewidth=1, alpha=0.5)
plt.plot(x_range, cqr_upper[:subset], 'r--', linewidth=1, alpha=0.5, label='LGBM+CQR')
plt.title(f'Uncertainty Quantification: HAQR (Scale-Up) vs CQR')
plt.legend()
plt.savefig('../results/experiment_02_uncertainty_plot.png', dpi=300)
print("  - 그래프 저장 완료.")
