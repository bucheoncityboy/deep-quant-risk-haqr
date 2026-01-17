# =============================================================================
# [File] 01_Validation_SOTA.py
# 내용: HAQR (Scale-Up) vs LGBM (Tuned) - The Ultimate Fair Match
# 설정:
#   1. HAQR: h_dim=32, epochs=100 (Capacity 확보)
#   2. LGBM: n_estimators=500, lr=0.05, subsample=0.8 (Strong Baseline)
#   3. Data: Raw Data (No Leakage)
#   4. N=100
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

sys.path.append(os.path.abspath('..'))

from src.data_gen import dual_regime, prep_data, denoise_wavelet
from src.models import build_haqr_model, pinball_loss
from src.utils import quantile_loss_stats, add_strat_metrics, QUANTILES

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import logging

tf.get_logger().setLevel(logging.ERROR)

# --- 설정 ---
NUM_SIMULATIONS = 100
SAVE_DIR_WEIGHTS = '../weights'
SAVE_DIR_RESULTS = '../results'

os.makedirs(SAVE_DIR_WEIGHTS, exist_ok=True)
os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)

# --- 전략 함수 ---
def calculate_m3_strategy(pred_quantiles, actual_returns, threshold):
    q05, q50, q95 = pred_quantiles[:, 0], pred_quantiles[:, 1], pred_quantiles[:, 2]
    signal = np.sign(q50)
    uncertainty = q95 - q05
    size = np.where(uncertainty > threshold, 0.5, 1.0)
    return signal * size * actual_returns

# --- 시뮬레이션 ---
all_results = []

print(f"[Start] 최종 공정 검증 (총 {NUM_SIMULATIONS}회)")
print("  - HAQR: Scale-Up (h_dim=32)")
print("  - LGBM: Tuned (n_est=500, lr=0.05, bagging=0.8)")

for z in tqdm(range(NUM_SIMULATIONS), desc="Simulating"):
    
    # 1. Data
    data = dual_regime(total_steps=5000, prob_switch=0.20, stdev=0.0145)
    model_data, data_raw = prep_data(data=data, with_flags=True)
    
    if len(model_data) < 100: continue

    split_idx = int(len(model_data) * 0.6)
    train_df = model_data.iloc[:split_idx]
    test_df = model_data.iloc[split_idx:]
    
    test_returns_filtered = data_raw.loc[test_df.index]['rets']
    start_date, end_date = test_df.index[0], test_df.index[-1]
    test_returns_full = data_raw.loc[start_date:end_date]['rets']
    
    features_trend = ['rets', 'rets2', 'rets3']
    features_market = ['regime', 'historical_vol']
    features_all = features_trend + features_market
    
    y_train = train_df['target_rets'].values
    y_test = test_df['target_rets'].values

    # 2. Preprocessing (Raw Data)
    X_train_trend = denoise_wavelet(train_df[features_trend].values)
    X_test_trend = denoise_wavelet(test_df[features_trend].values)
    X_train_market = train_df[features_market].values
    X_test_market = test_df[features_market].values
    
    sc_trend = StandardScaler().fit(X_train_trend)
    X_train_trend_sc = sc_trend.transform(X_train_trend)
    X_test_trend_sc = sc_trend.transform(X_test_trend)
    
    sc_market = StandardScaler().fit(X_train_market)
    X_train_market_sc = sc_market.transform(X_train_market)
    X_test_market_sc = sc_market.transform(X_test_market)
    
    X_train_lgbm = train_df[features_all].values
    X_test_lgbm = test_df[features_all].values
    sc_lgbm = StandardScaler().fit(X_train_lgbm)
    X_train_lgbm_sc = sc_lgbm.transform(X_train_lgbm)
    X_test_lgbm_sc = sc_lgbm.transform(X_test_lgbm)

    sc_y = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_sc = sc_y.transform(y_train.reshape(-1, 1))

    # 3. LGBM (Tuned Version - 공정한 강적)
    lgbm_preds_test = []
    lgbm_preds_train = []
    for q in QUANTILES:
        # [수정] LGBM 하이퍼파라미터 강화
        model = LGBMRegressor(
            objective='quantile', 
            alpha=q, 
            n_estimators=500,      # 트리 개수 증가 (기본 100)
            learning_rate=0.05,    # 학습률 감소로 정교화 (기본 0.1)
            num_leaves=31,         # 복잡도 유지
            subsample=0.8,         # 배깅 (Overfitting 방지)
            colsample_bytree=0.8,  # 피처 샘플링
            random_state=z, 
            verbose=-1
        )
        model.fit(X_train_lgbm_sc, y_train)
        lgbm_preds_test.append(model.predict(X_test_lgbm_sc))
        lgbm_preds_train.append(model.predict(X_train_lgbm_sc))
    pred_lgbm_test = np.vstack(lgbm_preds_test).T
    pred_lgbm_train = np.vstack(lgbm_preds_train).T

    # 4. HAQR (Scale-Up Version)
    tf.keras.backend.clear_session()
    haqr_model, _, _ = build_haqr_model(
        num_trend_features=len(features_trend),
        num_market_features=len(features_market),
        h_dim=32  # Scale-Up
    )
    haqr_model.compile(optimizer=Adam(0.001), loss=pinball_loss)
    
    haqr_model.fit(
        {'trend_input': X_train_trend_sc, 'market_input': X_train_market_sc},
        y_train_sc,
        epochs=100,     # Scale-Up
        batch_size=64,  # Scale-Up
        verbose=0
    )
    
    pred_haqr_test = sc_y.inverse_transform(haqr_model.predict({'trend_input': X_test_trend_sc, 'market_input': X_test_market_sc}, verbose=0))
    pred_haqr_train = sc_y.inverse_transform(haqr_model.predict({'trend_input': X_train_trend_sc, 'market_input': X_train_market_sc}, verbose=0))

    # 5. Evaluate
    row_lgbm_loss = quantile_loss_stats(y_test, pred_lgbm_test, 'lgbm')
    row_haqr_loss = quantile_loss_stats(y_test, pred_haqr_test, 'haqr')
    
    th_lgbm = np.mean(pred_lgbm_train[:, 2] - pred_lgbm_train[:, 0])
    th_haqr = np.mean(pred_haqr_train[:, 2] - pred_haqr_train[:, 0])
    
    strat_lgbm = calculate_m3_strategy(pred_lgbm_test, test_returns_filtered.values, th_lgbm)
    strat_haqr = calculate_m3_strategy(pred_haqr_test, test_returns_filtered.values, th_haqr)
    
    final_row = pd.concat([row_lgbm_loss, row_haqr_loss], axis=1)
    final_row = add_strat_metrics(final_row, test_returns_full, 'bah') 
    final_row = add_strat_metrics(final_row, pd.Series(strat_lgbm), 'lgbm_m3')
    final_row = add_strat_metrics(final_row, pd.Series(strat_haqr), 'haqr_m3')
    
    all_results.append(final_row)

    if z == NUM_SIMULATIONS - 1:
        haqr_model.save_weights(os.path.join(SAVE_DIR_WEIGHTS, 'haqr_best_model.weights.h5'))

# --- Result ---
final_report = pd.concat(all_results, ignore_index=True)
csv_path = os.path.join(SAVE_DIR_RESULTS, 'experiment_01_sota_revised.csv')
final_report.to_csv(csv_path, index=False)

print(f"\n[최종 검증 완료] Tuned LGBM vs Scale-Up HAQR (N={NUM_SIMULATIONS})")
print(f"1. Pinball Loss: LGBM {final_report['lgbm_pinball_loss'].mean():.6f} vs HAQR {final_report['haqr_pinball_loss'].mean():.6f}")
print(f"2. Sharpe Ratio: LGBM {final_report['lgbm_m3_sr'].mean():.4f} vs HAQR {final_report['haqr_m3_sr'].mean():.4f}")
