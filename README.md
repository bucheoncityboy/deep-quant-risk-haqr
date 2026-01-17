HAQR: Hierarchical Attention Quantile RegressionRisk-Aware Meta-Labeling & M3 Position Sizing Model📌 Abstract**HAQR(Hierarchical Attention Quantile Regression)**은 금융 시계열 데이터의 '분류 병목 현상(Classification Bottleneck)'을 해결하기 위해 제안된 딥러닝 아키텍처입니다. 기존의 메타 라벨링(Meta-Labeling)이 진입 여부(M2)만을 이진 분류하여 리스크 정보를 소실했던 한계를 극복하고, **계층적 어텐션(Hierarchical Attention)**과 **분위수 회귀(Quantile Regression)**를 결합하여 최적의 베팅 크기(M3)를 산출합니다1111.이중 국면(Dual Regime) 시뮬레이션 결과, HAQR은 SOTA 모델인 Tuned LightGBM과 최신 파운데이션 모델 Lag-Llama를 압도하는 성과를 보였으며, 특히 Sharpe Ratio 측면에서 4배 이상의 자본 효율성을 입증했습니다2.🚀 Key FeaturesHierarchical Attention Mechanism: 금융 데이터의 노이즈를 제어하기 위해 Trend Group(추세)과 Market Group(시장 국면)을 의미론적으로 분리하여 계층적으로 처리합니다3333.End-to-End M3 Sizing: 단순한 방향 예측(Point Prediction)을 넘어, 수익률의 조건부 분포(Conditional Distribution)를 직접 예측하여 진입 결정(M2)과 포지션 사이징(M3)을 통합했습니다4.Risk-Aware Quantile Head: Pinball Loss를 기반으로 꼬리 리스크(Tail Risk)를 정량화하며, Non-crossing 구조를 통해 분위수 역전 현상을 방지합니다5555.Explainable AI (XAI): F-Fidelity 프레임워크 검증을 통해, 모델의 어텐션 가중치가 실제 예측에 유의미하게 기여함을 입증했습니다6666.🏗 Model ArchitectureHAQR은 입력 데이터를 성격에 따라 그룹화하고, 팩터 레벨(Factor-level)과 그룹 레벨(Group-level)의 어텐션을 거쳐 최종 분포를 예측하는 구조를 가집니다7777.1. Semantic Grouping & Hierarchical Attention입력 데이터(X)는 두 가지 그룹으로 나뉩니다8:Trend Group: 자산의 과거 수익률 등 모멘텀 정보Market Group: 시장 국면(Regime), 변동성 등 거시 정보Attention Formula (Unicode):Intra-Group (Factor Level):h_group = Σ α_i · (W_e · x_i)Inter-Group (Group Level):c_final = Σ β_group · h_group(여기서 α와 β는 각 단계의 어텐션 가중치를 의미함) 99992. Quantile Head (Output)최종 컨텍스트 벡터(c_final)를 통해 3가지 분위수(Q=0.05, 0.50, 0.95)를 예측합니다.Left Tail (0.05): 최악의 상황 (Downside Risk)Median (0.50): 기대 수익률Right Tail (0.95): 최상의 상황 (Upside Potential)Objective Function (Pinball Loss):L(y, ŷ, τ) = max(τ(y - ŷ), (τ - 1)(y - ŷ)) 10📊 Performance이중 국면(Dual Regime) AR(3) 데이터를 이용한 100회 몬테카를로 시뮬레이션 결과입니다11111111.ModelPinball Loss (Risk)Sharpe Ratio (Efficiency)비고HAQR (Proposed)0.0032425.2852분포 기반 사이징 적용Tuned LightGBM0.0035011.3434SOTA Tabular ModelLag-Llama0.003853-Zero-shot Foundation ModelResult: HAQR은 불확실성이 높은 구간(High Volatility Regime)에서 예측 범위를 넓히고 포지션을 축소하여(Intrinsic Sizing), 거래 비용을 고려하더라도 압도적인 경제적 성과를 달성했습니다12121212.🛠 UsagePrerequisitesPython 3.8+PyTorch, NumPy, Pandas, Scikit-learnInstallationBashgit clone https://github.com/your-username/HAQR.git
cd HAQR
pip install -r requirements.txt
Configuration (Hyperparameters)본 연구는 과도한 튜닝 없이도 강건함을 보이기 위해 표준적인 파라미터를 사용했습니다13.Python# config.py
CONFIG = {
    "hidden_dim": 32,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "quantiles": [0.05, 0.50, 0.95]
}
Running the ModelPythonfrom model import HAQR
from data_loader import DualRegimeDataset

# 1. 데이터 로드
dataset = DualRegimeDataset(n_samples=10000)

# 2. 모델 초기화
model = HAQR(input_dim=dataset.input_dim, hidden_dim=32)

# 3. 학습 및 추론
# (Code snippet for training loop with Pinball Loss...)
📂 Directory StructureHAQR/
├── data/
│   ├── generate_data.py    # AR(3) Dual Regime 시뮬레이션 데이터 생성 [cite: 112]
│   └── processing.py       # Semantic Grouping (Trend/Market) 전처리
├── models/
│   ├── haqr.py             # HAQR 메인 아키텍처 (Encoder, Attention, Head)
│   └── layers.py           # Custom Layers (Non-crossing Quantile Layer)
├── utils/
│   ├── loss.py             # Pinball Loss 구현 [cite: 107]
│   └── metrics.py          # Sharpe Ratio, PICP, MPIW 계산
├── train.py                # 학습 루프
├── evaluate.py             # 백테스팅 및 결과 시각화
├── README.md
└── requirements.txt
📜 References이 프로젝트는 다음 연구들을 기반으로 수행되었습니다.Kim, J. (2026). HAQR: Hierarchical Attention for Risk Quantification and M3 Position Sizing. QuantLab Final Report. 14141414López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. (Meta-labeling concepts) 15Joubert, J. F. (2022). Meta-Labeling: Theory and Framework. (Simulation data design) 16Zheng, X., et al. (2025). F-Fidelity: A Robust Framework for Faithfulness Evaluation. (XAI evaluation) 17Author: Jaewon Kim (QuantLab)Contact: [Your Email / LinkedIn]
