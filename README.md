# HAQR: Hierarchical Attention Quantile Regression

<p align="center">
  <b>금융 시계열 예측을 위한 계층적 어텐션 기반 분위수 회귀 모델</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 📖 Overview

HAQR(Hierarchical Attention Quantile Regression)는 **금융 시계열의 불확실성을 정량화**하기 위해 설계된 딥러닝 모델입니다. 기존의 점 예측(point prediction) 방식을 넘어, **분위수 회귀(Quantile Regression)**를 통해 예측 구간을 제공하며, **계층적 어텐션 메커니즘(Hierarchical Attention Network)**을 활용하여 해석 가능한 예측을 수행합니다.

### 🎯 핵심 기여 (Key Contributions)

| 기여 | 설명 |
|------|------|
| **1. Hierarchical Attention Network** | 2단계 어텐션 구조 (Factor-Level → Group-Level)를 통해 피처 중요도를 계층적으로 학습 |
| **2. Non-Crossing Quantile Head** | Softplus 기반 델타 구조로 분위수 교차 문제를 원천적으로 해결 (Q5 < Q50 < Q95 보장) |
| **3. Intrinsic Uncertainty Quantification** | 별도의 교정(calibration) 없이 모델 자체에서 불확실성 구간을 직접 출력 |
| **4. Explainable AI (XAI)** | F-Fidelity 검증을 통한 어텐션 가중치의 충실성(Faithfulness) 입증 |

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────┐
                    │        HAQR Architecture         │
                    └──────────────────────────────────┘
                                    │
                              [Input Layer]
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
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
              │                                         │
              └────────────────┬────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Group-Level         │
                    │ Attention           │
                    │ (Trend vs Market)   │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Context Vector    │
                    └─────────┬───────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │   Non-Crossing Quantile Head      │
              │                                   │
              │  Q(0.05) ──→ Q(0.50) ──→ Q(0.95)  │
              │        +δ₁        +δ₂             │
              │     (softplus)  (softplus)        │
              └───────────────────────────────────┘
```

---

## 📁 Project Structure

```
HAQR/
├── src/                          # 핵심 소스 코드
│   ├── models.py                 # HAQR 모델 정의 (HAN + Quantile Head)
│   ├── data_gen.py               # 시뮬레이션 데이터 생성 (Dual-Regime AR Process)
│   └── utils.py                  # 유틸리티 함수 (PSR, DSR, MDD 계산 등)
│
├── experiment/                   # 실험 스크립트
│   ├── 01_Validation_SOTA.py     # SOTA 비교 실험 (HAQR vs LGBM)
│   ├── 01_B_Validation_LagLlama.py # Lag-Llama 비교 실험
│   ├── 02_Validation_Uncertainty.py # 불확실성 정량화 검증 (PICP, MPIW)
│   ├── 03_Validation_XAI.py      # 설명가능성 검증 (F-Fidelity)
│   ├── 04_Validation_Economic.py # 경제적 성과 분석 (PSR, DSR)
│   └── 05_Validation_Ablation.py # Ablation Study (Monte Carlo)
│
└── README.md

# 실험 실행 시 자동 생성되는 폴더:
# - weights/   : 학습된 모델 가중치 (.h5)
# - results/   : 실험 결과 (CSV, PNG)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/HAQR.git
cd HAQR

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

### 3. Run Experiments

```bash
# Navigate to experiment directory
cd experiment

# 1. SOTA Comparison (HAQR vs LGBM)
python 01_Validation_SOTA.py

# 2. Uncertainty Quantification
python 02_Validation_Uncertainty.py

# 3. XAI Validation (F-Fidelity)
python 03_Validation_XAI.py

# 4. Economic Performance Analysis
python 04_Validation_Economic.py

# 5. Ablation Study
python 05_Validation_Ablation.py
```

---

## 📊 Experimental Results

### 1. SOTA Comparison (N=100 Simulations)

| Model | Pinball Loss ↓ | Sharpe Ratio ↑ | PSR (vs 0) ↑ | DSR (vs N) ↑ |
|-------|---------------|----------------|--------------|--------------|
| LGBM (Tuned) | 0.00612 | 0.82 | 0.71 | 0.32 |
| **HAQR (Proposed)** | **0.00584** | **1.05** | **0.78** | **0.45** |

### 2. Uncertainty Quantification (90% Prediction Interval)

| Model | PICP (Target: 0.90) ↑ | MPIW (Width) ↓ |
|-------|----------------------|----------------|
| LGBM (Raw) | 0.78 | 0.041 |
| LGBM + CQR | 0.91 | 0.062 |
| **HAQR (Intrinsic)** | **0.89** | **0.048** |

### 3. XAI Validation (F-Fidelity Score)

- **MoRF (Most Relevant First)**: 중요 피처 제거 시 Loss 급상승 ✓
- **LeRF (Least Relevant First)**: 비중요 피처 제거 시 Loss 유지 ✓
- **F-Fidelity Score**: MoRF - LeRF > 0 (유의미한 차이 확인)

---

## 🔧 Model Configuration

### Key Hyperparameters

```python
# models.py
QUANTILES = [0.05, 0.5, 0.95]  # 예측할 분위수 (90% 신뢰구간)

# build_haqr_model()
h_dim = 32       # Hidden dimension (Scale-Up 버전)
epochs = 100     # 학습 에폭
batch_size = 64  # 배치 크기
```

### Loss Function

```python
def pinball_loss(y_true, y_pred):
    """Quantile Regression Loss (Pinball Loss)"""
    q = tf.constant(np.array(QUANTILES, dtype=np.float32).reshape(1, -1))
    e = y_true - y_pred
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
```

---

## 📈 Trading Strategy (M3 Sizing)

HAQR 모델의 분위수 출력을 활용한 **불확실성 인지 포지션 사이징** 전략:

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

### Risk Metrics

| Metric | Description |
|--------|-------------|
| **PSR (Probabilistic Sharpe Ratio)** | Sharpe Ratio가 0보다 클 확률 |
| **DSR (Deflated Sharpe Ratio)** | 다중 시도(N=100)를 보정한 Sharpe Ratio |
| **MDD (Maximum Drawdown)** | 최대 낙폭 |

---

## 🧪 Data Generation

본 연구에서는 **Dual-Regime AR(3) Process**를 사용하여 시뮬레이션 데이터를 생성합니다:

```python
# Regime 1 (Normal): φ = (0.25, -0.20, 0.35)
# Regime 2 (Crisis): φ = (-0.25, 0.20, -0.35)

data = dual_regime(
    total_steps=5000,    # 총 데이터 포인트
    prob_switch=0.20,    # 레짐 전환 확률
    stdev=0.0145         # 노이즈 표준편차
)
```

---

## 📝 Citation

```bibtex
@article{haqr2024,
  title={HAQR: Hierarchical Attention Quantile Regression for Financial Time Series Forecasting},
  author={Kim, Jaewon},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

- **Author**: Kim Jaewon
- **Email**: [your-email@example.com]

---

<p align="center">
  Made with ❤️ for Quantitative Finance Research
</p>
