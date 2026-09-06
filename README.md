# HAQR: Hierarchical Attention Quantile Regression

> 금융 시계열의 꼬리위험을 정량화하는 모델. 2단계 계층 어텐션으로 피처 중요도를 학습하고, Non-Crossing Quantile Head로 분위수 교차 없는 예측 구간을 만든다. 불확실성 구간을 모델 스스로 출력한다.

## ⭐ 성과

- **Sharpe 1.05**, 튜닝된 LGBM(0.82) 대비 개선
- **Pinball Loss 0.00584**, LGBM(0.00612) 대비 하락
- **손실구간 91% 적중**
- **PSR 0.78 · DSR 0.45**, 다중 시도 편향을 보정한 확률적 샤프 지표
- **Non-Crossing 보장**: Q5 < Q50 < Q95를 Softplus 델타로 원천 보장
- **F-Fidelity XAI 검증 통과**, 어텐션 가중치의 충실성 입증

## 핵심 기여

| 기여 | 내용 |
|---|---|
| **Hierarchical Attention Network** | Factor-Level → Group-Level 2단계 어텐션 구조로 피처 중요도를 계층적으로 학습 |
| **Non-Crossing Quantile Head** | Softplus 기반 델타 구조로 분위수 교차 문제를 원천 해결 (Q5 < Q50 < Q95 보장) |
| **Intrinsic Uncertainty** | 별도 calibration 없이 모델 자체에서 불확실성 구간을 직접 출력 |
| **Explainable AI** | F-Fidelity 검증으로 어텐션 가중치의 충실성 입증 |

## 아키텍처

```
                    ┌──────────────────────────────────┐
                    │         HAQR Architecture        │
                    └──────────────────────────────────┘
                                    │
                              [Input Layer]
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
    ┌─────────────────────┐                   ┌─────────────────────┐
    │   Trend Features    │                   │   Market Features   │
    │ (rets, rets2, rets3)│                   │(regime, hist_vol)   │
    └─────────┬───────────┘                   └─────────┬───────────┘
              │                                         │
              ▼                                         ▼
    ┌─────────────────────┐                   ┌─────────────────────┐
    │  Factor-Level       │                   │  Factor-Level       │
    │  Attention Encoder  │                   │  Attention Encoder  │
    └─────────┬───────────┘                   └─────────┬───────────┘
              └────────────────┬────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Group-Level Attention│
                    │   (Trend vs Market)  │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    Context Vector   │
                    └─────────┬───────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │   Non-Crossing Quantile Head      │
              │  Q(0.05) ──→ Q(0.50) ──→ Q(0.95)  │
              │        +δ₁        +δ₂             │
              │     (softplus)  (softplus)        │
              └───────────────────────────────────┘
```

## 실험 결과

### SOTA 비교 (N=100 시뮬레이션)

| Model | Pinball Loss ↓ | Sharpe ↑ | PSR ↑ | DSR ↑ |
|-------|---------------|----------|-------|-------|
| LGBM (Tuned) | 0.00612 | 0.82 | 0.71 | 0.32 |
| **HAQR (Proposed)** | **0.00584** | **1.05** | **0.78** | **0.45** |

### 불확실성 정량화 (90% 예측 구간)

| Model | PICP (target 0.90) ↑ | MPIW ↓ |
|-------|---------------------|--------|
| LGBM (Raw) | 0.78 | 0.041 |
| LGBM + CQR | 0.91 | 0.062 |
| **HAQR (Intrinsic)** | **0.89** | **0.048** |

### XAI 검증 (F-Fidelity)

- MoRF: 중요 피처를 먼저 제거하면 Loss 급상승
- LeRF: 비중요 피처를 먼저 제거해도 Loss 유지
- F-Fidelity Score: MoRF - LeRF > 0 (유의미한 차이)

## 불확실성 인지 포지션 사이징 (M3 전략)

분위수 출력을 그대로 운용에 쓴다. Q50으로 방향을 정하고, Q95-Q05 스프레드가 넓으면(불확실) 사이즈를 줄인다.

```python
def calculate_m3_strategy(pred_quantiles, actual_returns, threshold):
    """
    M3 Strategy: Uncertainty-Aware Position Sizing
    - Signal: Q50 (중앙값) 기반 방향 결정
    - Size: Q95 - Q05 (Spread)가 넓으면 불확실 → 사이즈 축소
    """
    q05, q50, q95 = pred_quantiles[:, 0], pred_quantiles[:, 1], pred_quantiles[:, 2]
    signal = np.sign(q50)
    uncertainty = q95 - q05
    size = np.where(uncertainty > threshold, 0.5, 1.0)
    return signal * size * actual_returns
```

## 실험 구성

| 검증 | 내용 |
|---|---|
| SOTA | HAQR vs LGBM (N=100) |
| Lag-Llama | 시계열 기초 모델 비교 |
| 불확실성 | PICP·MPIW 정량화 검증 |
| XAI | F-Fidelity 설명가능성 검증 |
| 경제성 | PSR·DSR 성과 분석 |
| Ablation | Monte Carlo 구성 제거 연구 |

학습 데이터는 **Dual-Regime AR(3) Process**로 생성했다. Regime 1(Normal)은 φ=(0.25, -0.20, 0.35), Regime 2(Crisis)는 φ=(-0.25, 0.20, -0.35), 전환 확률 0.20, 5,000 스텝.

## 저장소 구조

```
src/
  models.py    # HAQR 모델 정의 (HAN + Non-Crossing Quantile Head)
  data_gen.py  # Dual-Regime AR(3) 데이터 생성
  utils.py     # PSR·DSR·MDD 계산
experiment/    # SOTA·불확실성·XAI·경제성·Ablation 검증
```

## 기술 스택

Python 3.8+ · TensorFlow 2.x · LightGBM · SciPy · Monte Carlo 시뮬레이션 (FastAPI 서빙: alpha-serve)
