# src/utils.py (DSR 버그 수정)

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, norm
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
import tensorflow.keras.backend as K

# 상수 설정
QUANTILES = [0.05, 0.5, 0.95]
EULER_GAMMA = 0.5772156649015329
NUM_TRIALS = 100  

def classification_stats(actual, predicted_binary, prefix, get_specificity, **kwargs):
    report = classification_report(actual, predicted_binary, output_dict=True, labels=[0, 1], zero_division=0)
    metrics_dict = report.get('1', {})
    
    predicted_score = kwargs.get('predicted_score', None)
    if predicted_score is None:
        predicted_score = predicted_binary 

    metrics_dict[prefix + '_accuracy'] = report.get('accuracy', 0)
    try:
        metrics_dict[prefix + '_auc'] = roc_auc_score(actual, predicted_score)
    except ValueError:
        metrics_dict[prefix + '_auc'] = 0.5
        
    row = pd.DataFrame.from_dict(metrics_dict, orient='index').T
    row = row.rename(columns={'precision': prefix + '_precision', 'recall': prefix + '_recall', 
                              'f1-score': prefix + '_f1_score', 'support': prefix + '_support'})
    if get_specificity:
        row[prefix + '_specificity'] = report.get('0', {}).get('recall', 0)
    return row

def calculate_quantile_loss(y_true, y_pred, quantiles):
    q = tf.constant(np.array(quantiles, dtype=np.float32).reshape(1, -1))
    e = y_true - y_pred
    loss = K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
    return loss.numpy()

def quantile_loss_stats(actual, predicted_quantiles, prefix):
    metrics_dict = {}
    actual_replicated = K.repeat_elements(tf.constant(actual.reshape(-1, 1), dtype=tf.float32), len(QUANTILES), axis=1)
    
    raw_loss = calculate_quantile_loss(actual_replicated, predicted_quantiles, QUANTILES)
    metrics_dict[prefix + '_pinball_loss'] = np.mean(raw_loss)
    
    for i, q in enumerate(QUANTILES):
        q_actual = actual_replicated[:, i:i+1]
        q_pred = predicted_quantiles[:, i:i+1]
        q_loss = calculate_quantile_loss(q_actual, q_pred, [q])
        metrics_dict[f'{prefix}_q{q}_loss'] = np.mean(q_loss)
        
    row = pd.DataFrame.from_dict(metrics_dict, orient='index').T
    return row

def get_expected_max_sr_z(T, N):
    """[DSR 보조] N번 시도 시 기대되는 최대 Z-score 계산"""
    if T < 100: return 0
    # E[max(Z)] 근사식 (Lopez de Prado)
    Z_inv_N = norm.ppf(1 - 1 / N)
    Z_inv_Ne = norm.ppf(1 - 1 / (N * np.e))
    E_max_Z = (1 - EULER_GAMMA) * Z_inv_N + EULER_GAMMA * Z_inv_Ne
    return E_max_Z

def get_psr(returns_series, benchmark_sr=0):
    returns = returns_series.dropna()
    T = len(returns)
    if T < 100 or returns.std() == 0: return 0.0

    sr = returns.mean() / returns.std()
    sr_skew = skew(returns)
    sr_kurt = kurtosis(returns, fisher=False) 
    std_sr = np.sqrt((1 + 0.5 * sr**2 * (sr_kurt - 1) - sr * sr_skew) / (T - 1))
    
    if std_sr == 0 or np.isnan(std_sr): return 0.0
    psr_z = (sr - benchmark_sr) / (std_sr + 1e-10)
    return norm.cdf(psr_z)

def calculate_mdd(returns_series):
    cumulative_returns = (1 + returns_series.dropna()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

def add_strat_metrics(row, rets, prefix):
    if rets.empty or rets.std() == 0 or pd.isna(rets.std()) or len(rets) < 100:
        mean_ret, std, sharpe_ratio, mdd, psr_0, dsr_N = 0, 0, 0, 0, 0, 0
    else:
        mean_ret = rets.mean()
        std = rets.std()
        sharpe_ratio = mean_ret / std * np.sqrt(252)
        mdd = calculate_mdd(rets)
        
        T = len(rets)
        sr_non_annualized = mean_ret / std
        sr_skew = skew(rets)
        sr_kurt = kurtosis(rets, fisher=False)
        
        # Sharpe Ratio의 표준오차
        std_sr = np.sqrt((1 + 0.5 * sr_non_annualized**2 * (sr_kurt - 1) - sr_non_annualized * sr_skew) / (T - 1))
        
        # 1. PSR (Benchmark = 0)
        if std_sr > 0:
            psr_0 = norm.cdf((sr_non_annualized - 0) / std_sr)
        else:
            psr_0 = 0
        
        # 2. DSR (Benchmark = E[maxSR])
        expected_max_z = get_expected_max_sr_z(T, NUM_TRIALS)
        benchmark_sr_max = expected_max_z * std_sr
        
        if std_sr > 0:
            dsr_N = norm.cdf((sr_non_annualized - benchmark_sr_max) / std_sr)
        else:
            dsr_N = 0
        
    row[prefix + '_mean'] = mean_ret
    row[prefix + '_stdev'] = std
    row[prefix + '_sr'] = sharpe_ratio
    row[prefix + '_mdd'] = mdd
    row[prefix + '_psr_vs_0'] = psr_0
    row[prefix + '_dsr_vs_N'] = dsr_N
    return row
