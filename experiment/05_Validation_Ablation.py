# =============================================================================
# [File] 05_Validation_Ablation_MC.py
# 내용: Ablation Study (Monte Carlo Simulation)
# 수정 사항: 
# 1. KeyError 완벽 해결 (모델별 컬럼명 동적 탐색)
# 2. M3 사이징 로직 적용 (Quantile Spread vs Historical Vol)
# 3. 메모리 누수 방지 (clear_session)
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

sys.path.append(os.path.abspath('..'))
from src.data_gen import dual_regime, prep_data, denoise_wavelet
from src.models import build_haqr_model, pinball_loss
from src.utils import add_strat_metrics, quantile_loss_stats

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling1D, Reshape, Multiply, Softmax
from tensorflow.keras.models import Model
import logging
tf.get_logger().setLevel(logging.ERROR)

# =============================================================================
# [Configuration] 사용자 설정
# =============================================================================
N_ITERATIONS = 50   # 몬테카를로 반복 횟수 (시간에 따라 조절하세요)
EPOCHS = 50        # 반복 실행이므로 Epoch를 적절히 조절
BATCH_SIZE = 64
H_DIM = 32
# =============================================================================

# -------------------------------------------------------
# 1. Model Builders
# -------------------------------------------------------
def build_flat_model(num_features, h_dim=32):
    """Hierarchy가 없는 Flat Attention 모델"""
    inputs = Input(shape=(num_features,), name='flat_input')
    reshaped = Reshape((num_features, 1))(inputs)
    
    # Flat Attention
    x = Dense(h_dim, activation='relu')(reshaped)
    scores = Dense(1)(x)
    weights = Softmax(axis=1)(scores)
    weighted = Multiply()([x, weights])
    context = GlobalAveragePooling1D()(weighted)
    
    # Quantile Head
    base = Dense(h_dim, activation='relu')(context)
    q05 = Dense(1, name='q05')(base)
    q50 = Dense(1, name='q50')(base)
    q95 = Dense(1, name='q95')(base)
    output = Concatenate()([q05, q50, q95])
    
    return Model(inputs, output, name="Flat_HAQR")

def build_mse_haqr_model(num_trend, num_market, h_dim=32):
    """HAQR 구조지만 목적함수가 MSE(Point Prediction)인 모델"""
    input_trend = Input(shape=(num_trend,), name='trend_input')
    input_market = Input(shape=(num_market,), name='market_input')
    x = Concatenate()([input_trend, input_market])
    x = Dense(h_dim, activation='relu')(x)
    output = Dense(1, name='mse_output')(x)
    return Model(inputs={'trend_input': input_trend, 'market_input': input_market}, outputs=output)

# -------------------------------------------------------
# 2. Monte Carlo Simulation Loop
# -------------------------------------------------------
mc_results_loss = []
mc_results_econ = []
equity_curves = {'HAQR': [], 'MSE': []} # 누적 수익률 곡선 저장용

print(f"[{'='*20} Monte Carlo Simulation Start (N={N_ITERATIONS}) {'='*20}]")

for i in tqdm(range(N_ITERATIONS), desc="Simulation Progress"):
    # [Step 0] 메모리 누수 방지 (중요)
    K.clear_session()
    
    # [Step 1] 데이터 생성 (매번 새로운 Random Seed -> 강건성 검증)
    data = dual_regime(total_steps=5000, prob_switch=0.20, stdev=0.0145)
    model_data, data_raw = prep_data(data=data, with_flags=True)
    train_df, test_df = train_test_split(model_data, test_size=0.3, shuffle=False)
    
    test_returns = data_raw.loc[test_df.index]['rets'].values
    test_raw_series = data_raw.loc[test_df.index]['rets']

    # Preprocessing
    features_trend = ['rets', 'rets2', 'rets3']
    features_market = ['regime', 'historical_vol']

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

    X_train_flat = np.hstack([X_train_trend_sc, X_train_market_sc])
    X_test_flat = np.hstack([X_test_trend_sc, X_test_market_sc])

    y_train = train_df['target_rets'].values.reshape(-1, 1)
    sc_y = StandardScaler().fit(y_train)
    y_train_sc = sc_y.transform(y_train)
    y_test = test_df['target_rets'].values

    # [Step 2] 모델 학습
    # 1) HAQR (Main Model)
    haqr, _, _ = build_haqr_model(X_train_trend.shape[1], X_train_market.shape[1], h_dim=H_DIM)
    haqr.compile(optimizer=Adam(0.001), loss=pinball_loss)
    haqr.fit({'trend_input': X_train_trend_sc, 'market_input': X_train_market_sc}, 
             y_train_sc, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # 2) Flat (Architecture Ablation) - 조건을 맞추기 위해 clipnorm 없이 동일 설정
    flat_model = build_flat_model(num_features=X_train_flat.shape[1], h_dim=H_DIM)
    flat_model.compile(optimizer=Adam(0.001), loss=pinball_loss)
    flat_model.fit(X_train_flat, y_train_sc, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # 3) MSE (Objective Ablation)
    mse_model = build_mse_haqr_model(X_train_trend.shape[1], X_train_market.shape[1], h_dim=H_DIM)
    mse_model.compile(optimizer=Adam(0.001), loss='mse')
    mse_model.fit({'trend_input': X_train_trend_sc, 'market_input': X_train_market_sc}, 
                  y_train_sc, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # [Step 3] 평가 - Architecture (Loss)
    pred_haqr = sc_y.inverse_transform(haqr.predict({'trend_input': X_test_trend_sc, 'market_input': X_test_market_sc}, verbose=0))
    pred_flat = sc_y.inverse_transform(flat_model.predict(X_test_flat, verbose=0))

    loss_haqr = quantile_loss_stats(y_test, pred_haqr, 'HAQR')
    loss_flat = quantile_loss_stats(y_test, pred_flat, 'Flat')
    
    # [FIXED] 각 모델별로 존재하는 컬럼만 선택 (KeyError 방지)
    cols_haqr = [c for c in loss_haqr.columns if 'pinball_loss' in c]
    cols_flat = [c for c in loss_flat.columns if 'pinball_loss' in c]
    
    mc_results_loss.append({
        'Iteration': i,
        'HAQR_Loss': loss_haqr[cols_haqr].sum(axis=1).values[0],
        'Flat_Loss': loss_flat[cols_flat].sum(axis=1).values[0]
    })

    # [Step 4] 평가 - Economic (Strategy)
    target_vol = 0.012
    min_lev, max_lev = 0.2, 3.0
    
    # A) HAQR Strategy: Risk-Aware (Spread Sizing)
    # 논리: Q95-Q05 Spread가 넓으면 불확실 -> 사이즈 축소
    spread_haqr = pred_haqr[:, 2] - pred_haqr[:, 0]
    vol_implied = spread_haqr / 3.29 # 90% Conf Interval to Sigma
    size_haqr_raw = target_vol / (vol_implied + 1e-6)
    # [Fairness] Mean Leverage = 1.0 Normalization
    size_haqr = np.clip(size_haqr_raw / size_haqr_raw.mean(), min_lev, max_lev)
    strat_haqr = np.sign(pred_haqr[:, 1]) * size_haqr * test_returns

    # B) MSE Strategy: Risk-Ignorant (Historical Vol Sizing)
    # 논리: 불확실성 예측 불가 -> 과거 변동성 의존
    pred_mse = sc_y.inverse_transform(mse_model.predict({'trend_input': X_test_trend_sc, 'market_input': X_test_market_sc}, verbose=0)).flatten()
    vol_hist = test_raw_series.rolling(20).std().bfill().values
    size_mse_raw = target_vol / (vol_hist + 1e-6)
    # [Fairness] Mean Leverage = 1.0 Normalization
    size_mse = np.clip(size_mse_raw / size_mse_raw.mean(), min_lev, max_lev)
    strat_mse = np.sign(pred_mse) * size_mse * test_returns

    # Metrics 저장 (Prefix 'HAQR', 'MSE' 사용)
    row_haqr = add_strat_metrics({}, pd.Series(strat_haqr), 'HAQR')
    row_mse = add_strat_metrics({}, pd.Series(strat_mse), 'MSE')
    
    # [Check] add_strat_metrics가 반환하는 실제 Key 사용 ('HAQR_sr', 'MSE_sr' 등)
    mc_results_econ.append({
        'Iteration': i,
        'HAQR_SR': row_haqr.get('HAQR_sr', np.nan),
        'MSE_SR': row_mse.get('MSE_sr', np.nan),
        'HAQR_Return': row_haqr.get('HAQR_mean', 0) * 252, 
        'MSE_Return': row_mse.get('MSE_mean', 0) * 252,
        'HAQR_MDD': row_haqr.get('HAQR_mdd', np.nan),
        'MSE_MDD': row_mse.get('MSE_mdd', np.nan)
    })
    
    # Equity Curve 저장 (Spaghetti Plot용)
    equity_curves['HAQR'].append((1 + strat_haqr).cumprod())
    equity_curves['MSE'].append((1 + strat_mse).cumprod())

# -------------------------------------------------------
# 3. Aggregation & Visualization
# -------------------------------------------------------
print(f"\n[{'='*20} Results Aggregation {'='*20}]")

# 1) Loss Analysis (Architecture)
df_mc_loss = pd.DataFrame(mc_results_loss)
print("\n[Pinball Loss Stats (Lower is Better)]")
print(df_mc_loss[['HAQR_Loss', 'Flat_Loss']].describe().T[['mean', 'std', 'min', 'max']])

plt.figure(figsize=(8, 6))
# Error Bar plot to show Stability
sns.barplot(data=df_mc_loss[['HAQR_Loss', 'Flat_Loss']], errorbar='sd', capsize=.1, palette='viridis')
plt.title(f'Architecture Ablation (N={N_ITERATIONS}): Pinball Loss\n(Mean ± Std Dev)')
plt.ylabel('Total Pinball Loss')
plt.savefig('../results/mc_ablation_arch_loss.png')

# 2) Economic Analysis (Objective)
df_mc_econ = pd.DataFrame(mc_results_econ)
print("\n[Economic Performance Stats (Higher is Better)]")
print(df_mc_econ[['HAQR_SR', 'MSE_SR', 'HAQR_Return', 'MSE_Return']].describe().T[['mean', 'std']])

# Boxplot for Sharpe Ratio
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_mc_econ[['HAQR_SR', 'MSE_SR']], palette='coolwarm')
plt.title(f'Objective Ablation (N={N_ITERATIONS}): Sharpe Ratio Distribution')
plt.ylabel('Sharpe Ratio')
plt.savefig('../results/mc_ablation_econ_sr.png')

# 3) Equity Curve with Confidence Interval (Spaghetti Plot Logic)
plt.figure(figsize=(12, 7))

# Helper to plot mean and fill std
def plot_conf_interval(data_list, label, color):
    # data_list: list of arrays (length: time_steps) -> Stack into 2D array
    # 길이를 맞추기 위해 최소 길이 기준으로 자름 (혹시 모를 길이 불일치 방지)
    min_len = min([len(x) for x in data_list])
    arr = np.vstack([x[:min_len] for x in data_list])
    
    mean_curve = np.mean(arr, axis=0)
    std_curve = np.std(arr, axis=0)
    time_steps = range(len(mean_curve))
    
    # Plot Mean
    plt.plot(time_steps, mean_curve, label=f'{label} (Mean)', color=color, linewidth=2)
    # Fill Std (Confidence Interval)
    plt.fill_between(time_steps, mean_curve - std_curve, mean_curve + std_curve, 
                     color=color, alpha=0.15, label=f'{label} (±1 Std)')

plot_conf_interval(equity_curves['HAQR'], 'HAQR (Risk-Aware)', 'blue')
plot_conf_interval(equity_curves['MSE'], 'MSE (Risk-Ignorant)', 'gray')

plt.title(f'Cumulative Return: Monte Carlo Simulation (N={N_ITERATIONS})\nShaded Area represents ±1 Standard Deviation')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../results/mc_ablation_equity_curve.png')

# Save Raw Data
df_mc_loss.to_csv('../results/mc_ablation_loss_raw.csv', index=False)
df_mc_econ.to_csv('../results/mc_ablation_econ_raw.csv', index=False)

print("\n[INFO] 몬테카를로 시뮬레이션 완료. 결과 이미지 및 CSV 저장됨.")
