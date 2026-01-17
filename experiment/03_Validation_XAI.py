# =============================================================================
# [File] 03_Validation_XAI.py (Refined F-Fidelity Version)
# 내용: XAI 충실성 검증 - F-Fidelity (Fine-tuning 기반) 구현
# =============================================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('..'))
from src.data_gen import dual_regime, prep_data, denoise_wavelet
from src.models import build_haqr_model, pinball_loss

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

# =============================================================================
# 1. 데이터 준비 (Data Preparation) - [수정됨: float32 형변환 추가]
# =============================================================================
# 1-1. 데이터 생성
data = dual_regime(total_steps=5000, prob_switch=0.20, stdev=0.0145)
model_data, _ = prep_data(data=data, with_flags=True)
train_df, test_df = train_test_split(model_data, test_size=0.3, shuffle=False)

features_info = ['rets', 'rets2', 'rets3']
features_market = ['regime', 'historical_vol']

# 1-2. 데이터 가공
X_train_trend = denoise_wavelet(train_df[features_info].values)
X_train_market = train_df[features_market].values
y_train = train_df['target_rets'].values.reshape(-1, 1)

X_test_trend = denoise_wavelet(test_df[features_info].values)
X_test_market = test_df[features_market].values
y_test = test_df['target_rets'].values.reshape(-1, 1)

# 1-3. 스케일링 및 형변환 (여기가 핵심 수정 사항)
scaler_trend = StandardScaler().fit(X_train_trend)
# [FIX] .astype(np.float32)를 추가하여 TensorFlow 호환성 확보
X_train_trend_sc = scaler_trend.transform(X_train_trend).astype(np.float32)
X_test_trend_sc = scaler_trend.transform(X_test_trend).astype(np.float32)

scaler_market = StandardScaler().fit(X_train_market)
X_train_market_sc = scaler_market.transform(X_train_market).astype(np.float32)
X_test_market_sc = scaler_market.transform(X_test_market).astype(np.float32)

scaler_y = StandardScaler().fit(y_train)
y_train_sc = scaler_y.transform(y_train).astype(np.float32)
y_test_sc = scaler_y.transform(y_test).astype(np.float32)

y_true_tensor = tf.convert_to_tensor(y_test_sc, dtype=tf.float32)


# =============================================================================
# 2. 모델 로드 및 F-Fidelity Fine-tuning
# =============================================================================
# 2-1. 모델 빌드
haqr_model, att_model_group, att_model_factor = build_haqr_model(
    num_trend_features=X_test_trend.shape[1],
    num_market_features=X_test_market.shape[1],
    h_dim=32
)

# 2-2. Pre-trained Weights 로드
weights_path = '../weights/haqr_best_model.weights.h5'
if os.path.exists(weights_path):
    haqr_model.load_weights(weights_path)
    print("[INFO] Original Best Weights 로드 완료.")
else:
    raise FileNotFoundError("가중치 파일 없음 - 먼저 모델을 학습하세요.")

# -----------------------------------------------------------------------------
# [F-Fidelity Step] 모델 Fine-tuning
# -----------------------------------------------------------------------------
print("\n[F-Fidelity] Fine-tuning 시작 (Random Masking Adaptation)...")

def create_masked_dataset(trend, market, y, batch_size=64, mask_rate=0.1):
    def _mask_step(inputs, targets):
        trend_in, market_in = inputs['trend_input'], inputs['market_input']
        
        # Trend Masking (float32 * float32 연산으로 안전)
        mask_t = tf.random.uniform(tf.shape(trend_in)) > mask_rate
        trend_masked = trend_in * tf.cast(mask_t, tf.float32)
        
        # Market Masking
        mask_m = tf.random.uniform(tf.shape(market_in)) > mask_rate
        market_masked = market_in * tf.cast(mask_m, tf.float32)
        
        return {'trend_input': trend_masked, 'market_input': market_masked}, targets

    dataset = tf.data.Dataset.from_tensor_slices(
        ({'trend_input': trend, 'market_input': market}, y)
    )
    dataset = dataset.shuffle(1024).batch(batch_size).map(_mask_step)
    return dataset

# [FIX] float32로 변환된 데이터를 넘김
ft_dataset = create_masked_dataset(X_train_trend_sc, X_train_market_sc, y_train_sc)

haqr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=pinball_loss)
haqr_model.fit(ft_dataset, epochs=5, verbose=1)

print("[F-Fidelity] Fine-tuning 완료. 평가를 시작합니다.\n")

# -----------------------------------------------------------------------------
# [F-Fidelity Step] 모델 Fine-tuning (Random Masking 적용)
# 리뷰어 지적 사항: 마스킹된 데이터에 대해 모델을 적응시켜 OOD 문제를 해결해야 함
# -----------------------------------------------------------------------------
print("\n[F-Fidelity] Fine-tuning 시작 (Random Masking Adaptation)...")

# Fine-tuning용 데이터셋 생성 (배치 단위로 랜덤 마스킹 적용)
def create_masked_dataset(trend, market, y, batch_size=64, mask_rate=0.1):
    def _mask_step(inputs, targets):
        trend_in, market_in = inputs['trend_input'], inputs['market_input']
        
        # Trend Masking
        mask_t = tf.random.uniform(tf.shape(trend_in)) > mask_rate
        trend_masked = trend_in * tf.cast(mask_t, tf.float32)
        
        # Market Masking
        mask_m = tf.random.uniform(tf.shape(market_in)) > mask_rate
        market_masked = market_in * tf.cast(mask_m, tf.float32)
        
        return {'trend_input': trend_masked, 'market_input': market_masked}, targets

    dataset = tf.data.Dataset.from_tensor_slices(
        ({'trend_input': trend, 'market_input': market}, y)
    )
    dataset = dataset.shuffle(1024).batch(batch_size).map(_mask_step)
    return dataset

# Fine-tuning 수행
# - Learning Rate를 낮게 설정(1e-5)하여 기존 지식을 유지하며 적응
# - Epochs는 논문에 따라 적은 횟수(5회) 수행
ft_dataset = create_masked_dataset(X_train_trend_sc, X_train_market_sc, y_train_sc)
haqr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=pinball_loss)
haqr_model.fit(ft_dataset, epochs=5, verbose=1)

print("[F-Fidelity] Fine-tuning 완료. 평가를 시작합니다.\n")
# -----------------------------------------------------------------------------

# Baseline Loss (Fine-tuned 모델 기준)
base_pred = haqr_model.predict({'trend_input': X_test_trend_sc, 'market_input': X_test_market_sc}, verbose=0)
y_pred_tensor = tf.convert_to_tensor(base_pred, dtype=tf.float32)
base_loss = np.mean(pinball_loss(y_true_tensor, y_pred_tensor).numpy())
print(f"  - Fine-tuned Model Pinball Loss: {base_loss:.6f}")

# =============================================================================
# 3. 중요도 추출 및 검증 (Evaluation)
# =============================================================================
# 중요도 추출 (Fine-tuned 가중치 기반)
w_groups = att_model_group.predict({'trend_input': X_test_trend_sc, 'market_input': X_test_market_sc}, verbose=0)
w_trend_group = w_groups[:, 0, 0].reshape(-1, 1)
w_market_group = w_groups[:, 1, 0].reshape(-1, 1)

w_factors = att_model_factor.predict({'trend_input': X_test_trend_sc, 'market_input': X_test_market_sc}, verbose=0)
w_trend_factors = w_factors['trend_factors'][:, :, 0]
w_market_factors = w_factors['market_factors'][:, :, 0]

global_imp_trend = w_trend_factors * w_trend_group
global_imp_market = w_market_factors * w_market_group
global_importance = np.hstack([global_imp_trend, global_imp_market])

# 평가 함수 (Inference with Masking)
def get_masked_loss(mask_indices_batch):
    x_trend = X_test_trend_sc.copy()
    x_market = X_test_market_sc.copy()
    for i in range(len(x_trend)):
        for idx in mask_indices_batch[i]:
            if idx < 3: x_trend[i, idx] = 0
            else:       x_market[i, idx-3] = 0
    
    # Fine-tuned 모델은 0으로 마스킹된 데이터에 익숙하므로 OOD 문제가 완화됨
    pred = haqr_model.predict({'trend_input': x_trend, 'market_input': x_market}, verbose=0)
    y_pred_t = tf.convert_to_tensor(pred, dtype=tf.float32)
    return np.mean(pinball_loss(y_true_tensor, y_pred_t).numpy())

sorted_indices_desc = np.argsort(global_importance, axis=1)[:, ::-1]
results = []
k_steps = [1, 2, 3, 4]  # 전체 피처 수가 5개이므로 4까지만 테스트

for k in tqdm(k_steps, desc="F-Fidelity Test"):
    # MoRF (Most Relevant First): 중요한 것부터 제거 -> Loss 급상승 예상
    morf_idx = sorted_indices_desc[:, :k]
    loss_morf = get_masked_loss(morf_idx)
    
    # LeRF (Least Relevant First): 안 중요한 것부터 제거 -> Loss 유지 예상
    lerf_idx = sorted_indices_desc[:, -k:]
    loss_lerf = get_masked_loss(lerf_idx)
    
    # Random: 무작위 제거 (Baseline)
    rand_loss_avg = 0
    for _ in range(5):
        rand_idx = np.array([np.random.choice(5, k, replace=False) for _ in range(len(y_test))])
        rand_loss_avg += get_masked_loss(rand_idx)
    loss_rand = rand_loss_avg / 5
    
    results.append({'Sparsity (k)': k, 'Fid+ (MoRF)': loss_morf, 'Fid- (LeRF)': loss_lerf, 'Random': loss_rand})

# =============================================================================
# 4. 결과 저장 및 시각화
# =============================================================================
df_fid = pd.DataFrame(results)
df_fid['F-Fidelity Score'] = df_fid['Fid+ (MoRF)'] - df_fid['Fid- (LeRF)']
print("\n[F-Fidelity 검증 결과 (Fine-tuned)]")
print(df_fid)

# 결과 CSV 저장
if not os.path.exists('../results'):
    os.makedirs('../results')
df_fid.to_csv('../results/experiment_03_ffidelity.csv', index=False)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(df_fid['Sparsity (k)'], df_fid['Fid+ (MoRF)'], 'r-o', label='MoRF (Remove Important)')
plt.plot(df_fid['Sparsity (k)'], df_fid['Fid- (LeRF)'], 'b-s', label='LeRF (Remove Unimportant)')
plt.plot(df_fid['Sparsity (k)'], df_fid['Random'], 'k--', label='Random')
plt.axhline(y=base_loss, color='g', linestyle=':', label='Original Loss (Fine-tuned)')

plt.title('F-Fidelity Curve (Fine-tuned HAQR)')
plt.xlabel('Number of Removed Features (k)')
plt.ylabel('Pinball Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../results/experiment_03_ffidelity_plot.png')
print("[INFO] 결과 그래프 저장 완료: ../results/experiment_03_ffidelity_plot.png")
