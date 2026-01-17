# HAQR: Hierarchical Attention Quantile Regression
**Risk-Aware Meta-Labeling & M3 Position Sizing Model**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Abstract
[cite_start]**HAQR(Hierarchical Attention Quantile Regression)**은 금융 시계열 데이터의 '분류 병목 현상(Classification Bottleneck)'을 해결하기 위해 제안된 딥러닝 아키텍처입니다[cite: 9, 23].

[cite_start]기존의 메타 라벨링(Meta-Labeling)이 진입 여부(M2)만을 이진 분류하여 리스크 정보를 소실했던 한계를 극복하고, **계층적 어텐션(Hierarchical Attention)**과 **분위수 회귀(Quantile Regression)**를 결합하여 불확실성에 기반한 최적의 베팅 크기(M3)를 산출합니다[cite: 8, 9, 26].

[cite_start]이중 국면(Dual Regime) 시뮬레이션 결과, HAQR은 SOTA 모델인 Tuned LightGBM과 최신 파운데이션 모델 Lag-Llama를 압도하는 성과를 보였으며, 특히 **Sharpe Ratio 5.28**을 달성하여 대조군 대비 **4배 이상의 자본 효율성**을 입증했습니다[cite: 10, 11].

## 💡 Motivation: The Classification Bottleneck
[cite_start]전통적인 메타 라벨링(Meta-Labeling)은 2차 모델을 이진 분류기(Binary Classifier)로 설계함으로써 고차원의 리스크 정보를 단순한 확률값(0~1)으로 압축해버리는 문제가 있습니다[cite: 23, 25].

* [cite_start]**Problem:** "얼마나 베팅할 것인가(M3)"를 결정하는 데 필요한 변동성과 꼬리 리스크 정보가 소실됨[cite: 26].
* [cite_start]**Solution (HAQR):** 수익률의 **조건부 분포(Conditional Distribution)** 전체를 추정하여 '정보 복원(Information Restoration)'을 수행하고, 이를 포지션 사이징에 직접 연결[cite: 27, 28].

## 🏗 Model Architecture
[cite_start]HAQR은 입력 데이터를 성격에 따라 그룹화하고, 계층적 어텐션을 통해 노이즈를 제어한 뒤, Non-crossing Quantile Head를 통해 분포를 예측하는 End-to-End 모델입니다[cite: 47].

### 1. Semantic Grouping & Hierarchical Attention
[cite_start]금융 데이터의 이질적 특성(Idiosyncratic vs Systematic)을 반영하여 두 그룹으로 분리 처리합니다[cite: 94, 97].

* [cite_start]**Trend Group:** 자산의 과거 수익률 등 모멘텀 정보[cite: 85].
* [cite_start]**Market Group:** 시장 국면(Regime), 변동성 등 거시 정보[cite: 86].

**Attention Mechanism (Unicode Formulas):**

1.  [cite_start]**Factor-level (Intra-Group):** 각 그룹 내부의 중요 변수 식별[cite: 88].
    > h_group = Σ α_i · (W_e · x_i)
2.  [cite_start]**Group-level (Inter-Group):** 추세 vs 시장 정보 간의 신뢰도 동적 판단[cite: 97].
    > c_final = Σ β_group · h_group

### 2. Non-Crossing Quantile Head
[cite_start]최종 컨텍스트 벡터(c_final)를 통해 3가지 분위수를 예측하며, 분위수 역전(Crossing) 방지 구조를 적용했습니다[cite: 103].

* [cite_start]**Objective:** **Pinball Loss** 최소화[cite: 107].
    > L(y, ŷ, τ) = max(τ(y - ŷ), (τ - 1)(y - ŷ))
* **Outputs:**
    * [cite_start]`Q(0.05)`: Downside Risk (최악의 상황)[cite: 69].
    * [cite_start]`Q(0.50)`: Median Return (기대 수익)[cite: 73].
    * [cite_start]`Q(0.95)`: Upside Potential (최상의 상황)[cite: 76].

## 📊 Experiments & Results
[cite_start]이중 국면(Dual Regime) AR(3) 데이터를 이용한 100회 몬테카를로 시뮬레이션 비교 결과입니다[cite: 11, 113].

### 1. Predictive Performance
| Model | Pinball Loss (Risk Accuracy) | Sharpe Ratio (Economic Value) |
| :--- | :--- | :--- |
| **HAQR (Proposed)** | **0.003242** | **5.2852** |
| Tuned LightGBM | 0.003501 | 1.3434 |
| Lag-Llama (Zero-shot) | 0.003853 | N/A |

> [cite_start]**Analysis:** HAQR은 LGBM 대비 **7.4% 낮은 Loss**를 기록했으며, 불확실성이 높은 구간에서 포지션을 축소하는 내재적 사이징(Intrinsic Sizing)을 통해 압도적인 Sharpe Ratio를 달성했습니다[cite: 151, 153].

### 2. Uncertainty Quantification (UQ)
[cite_start]사후 보정(Calibration) 없이도 CQR(Conformalized Quantile Regression)과 대등하거나 더 우수한 불확실성 추정 능력을 보였습니다[cite: 171].

* [cite_start]**PICP (Coverage):** 91.48% (Target 90% 상회)[cite: 170].
* [cite_start]**MPIW (Width):** 0.0541 (LGBM+CQR 0.0548 대비 좁은 구간으로 더 예리한 예측)[cite: 170].

### 3. XAI Faithfulness
[cite_start]**F-Fidelity** 프레임워크 검증 결과, 중요 변수 제거(MoRF) 시 Loss가 급격히 상승하고 비중요 변수 제거(LeRF) 시 평탄한 모습을 보여, 모델의 설명력이 실제 예측에 충실함(Faithful)을 입증했습니다[cite: 223, 226].

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch, NumPy, Pandas, Scikit-learn

### Installation
```bash
git clone [https://github.com/your-username/HAQR.git](https://github.com/your-username/HAQR.git)
cd HAQR
pip install -r requirements.txt
