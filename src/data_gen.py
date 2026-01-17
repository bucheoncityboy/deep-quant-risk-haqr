# =============================================================================
# [File] src/data_gen.py
# 내용: 데이터 생성 (Raw Data 반환 버전 - Wavelet/EMA 모두 삭제)
# =============================================================================

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split

# --- 상수 설정 ---
R1_INIT, R2_INIT, R3_INIT = 0.032, 0.020, -0.042 
INNER_STEPS = 30
P1, P2, P3 = 0.25, -0.20, 0.35
PN1, PN2, PN3 = -0.25, 0.20, -0.35

def _gen_data(phi1, phi2, phi3, flag, stdev, drift, steps, r_state):
    r1, r2, r3 = r_state
    rets, flags = [], []
    for _ in range(steps):
        a = np.random.normal(loc=0, scale=stdev, size=1)
        rt = drift + phi1 * r1 + phi2 * r2 + phi3 * r3 + a
        flags.append(flag)
        rets.append(float(rt[0]))
        r3, r2, r1 = r2, r1, float(rt[0])
    return rets, flags, (r1, r2, r3)

def _gen_dual_regime(steps, inner_steps, prob_switch, stdev):
    rets, flags = [], []
    current_r = (R1_INIT, R2_INIT, R3_INIT)
    for _ in range(steps):
        is_regime_two = np.random.uniform() < prob_switch
        if is_regime_two: 
            r, f, new_r = _gen_data(PN1, PN2, PN3, 1, stdev, -0.0001, inner_steps, current_r)
        else: 
            r, f, new_r = _gen_data(P1, P2, P3, 0, stdev, 0.0, inner_steps, current_r)
        rets.extend(r)
        flags.extend(f)
        current_r = new_r
    return rets, flags

def dual_regime(total_steps, prob_switch, stdev):
    steps = int(total_steps / INNER_STEPS)
    rets, flags = _gen_dual_regime(steps, INNER_STEPS, prob_switch, stdev)
    dr = pd.date_range(end=dt.datetime.now(), periods=len(rets), freq='D', normalize=True)
    return pd.DataFrame({'rets': np.array(rets).flatten(), 'flags': flags}, index=dr)

def prep_data(data, with_flags, regime_lag=5):
    data['target'] = data['rets'].apply(lambda x: 0 if x < 0 else 1).shift(-1)
    data['target_rets'] = data['rets'].shift(-1)
    data['pmodel'] = data['rets'].apply(lambda x: 1 if x > 0.0 else 0)
    data['prets'] = data['pmodel'].shift(1) * data['rets']
    data['rets2'] = data['rets'].shift(1)
    data['rets3'] = data['rets'].shift(2)
    data['historical_vol'] = data['rets'].rolling(INNER_STEPS).std().shift(1) 
    if with_flags:
        data['regime'] = data['flags'].shift(regime_lag)
    data.dropna(inplace=True) 
    model_data = data[data['pmodel'] == 1].copy()
    return model_data, data

def denoise_wavelet(data, wavelet='db4', level=1):
    """
    [최종 수정] Raw Data (Identity)
    - Wavelet: 삭제 (미래 참조 문제)
    - EMA: 삭제 (반응 속도 지연 문제)
    - 결론: 원본 데이터를 그대로 반환하여 모델의 순수 성능 검증
    """
    return data
