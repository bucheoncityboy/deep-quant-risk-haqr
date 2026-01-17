# =============================================================================
# [Notebook] 04_Validation_Economic.ipynb
# 주제: [검증 4] 경제적 성과 심층 분석 (PSR, DSR, MDD Statistical Test)
# 수정: 컬럼명 불일치 해결 (_meta -> _m3)
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------------------------------------------------------
# 1. 데이터 로드
# -----------------------------------------------------------------------------
# [주의] 01번 Revised 스크립트의 결과 파일을 로드합니다.
RESULTS_PATH = '../results/experiment_01_sota_revised.csv'

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(f"결과 파일이 없습니다: {RESULTS_PATH}\n 먼저 '01_Validation_SOTA_Revised.ipynb'를 실행하세요.")

df = pd.read_csv(RESULTS_PATH)
print(f"데이터 로드 완료: {df.shape[0]}회의 시뮬레이션 결과")

# -----------------------------------------------------------------------------
# 2. 분석할 전략 및 지표 정의
# -----------------------------------------------------------------------------
# [수정] 01번 스크립트의 저장 이름과 일치시킴 (lgbm_m3, haqr_m3)
strategies = {
    'bah': 'Buy & Hold',
    'lgbm_m3': 'SOTA (LGBM)',
    'haqr_m3': 'HAQR (Proposed)'
}

# 분석할 핵심 경제 지표
metrics = {
    '_sr': 'Sharpe Ratio',
    '_mdd': 'Max Drawdown',
    '_psr_vs_0': 'Prob. Sharpe Ratio (PSR)',
    '_dsr_vs_N': 'Deflated Sharpe Ratio (DSR)'
}

# -----------------------------------------------------------------------------
# 3. 통계적 유의성 검정 (Statistical Significance Test)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print(" [통계 검정] HAQR vs SOTA (LGBM) : Paired t-test (One-sided)")
print("="*80)
print(f"{'Metric':<25} | {'HAQR Mean':<10} | {'LGBM Mean':<10} | {'Diff':<10} | {'p-value':<10} | {'Significance'}")
print("-" * 90)

for metric_suffix, metric_name in metrics.items():
    # [수정] _meta -> _m3 로 변경된 컬럼명 사용
    haqr_col = f'haqr_m3{metric_suffix}'
    lgbm_col = f'lgbm_m3{metric_suffix}'
    
    if haqr_col not in df.columns or lgbm_col not in df.columns:
        print(f"[Skip] {metric_name}: 컬럼을 찾을 수 없음 ({haqr_col})")
        continue

    haqr_vals = df[haqr_col]
    lgbm_vals = df[lgbm_col]
    
    # t-test 수행 (HAQR > LGBM 인지 검정)
    t_stat, p_val = stats.ttest_rel(haqr_vals, lgbm_vals, alternative='greater')
    
    mean_haqr = haqr_vals.mean()
    mean_lgbm = lgbm_vals.mean()
    diff = mean_haqr - mean_lgbm
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    print(f"{metric_name:<25} | {mean_haqr:<10.4f} | {mean_lgbm:<10.4f} | {diff:<10.4f} | {p_val:<10.4e} | {sig}")

print("-" * 90)
print("(*: p<0.05, **: p<0.01, ***: p<0.001)")

# -----------------------------------------------------------------------------
# 4. 결과 시각화 (Distribution Plots)
# -----------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Economic Performance Distribution (N={len(df)} Simulations)', fontsize=20)

axes_flat = axes.flatten()

for idx, (metric_suffix, metric_name) in enumerate(metrics.items()):
    ax = axes_flat[idx]
    
    # 데이터 준비 (Long Format)
    plot_data = pd.DataFrame()
    for strat_key, strat_label in strategies.items():
        col_name = f'{strat_key}{metric_suffix}'
        if col_name in df.columns:
            temp_df = pd.DataFrame({
                'Strategy': strat_label,
                'Value': df[col_name]
            })
            plot_data = pd.concat([plot_data, temp_df])
            
    # 바이올린 플롯
    sns.violinplot(data=plot_data, x='Strategy', y='Value', ax=ax, 
                   palette="muted", inner="quartile", alpha=0.6)
    
    ax.set_title(f'{metric_name} Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metric_name)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# -----------------------------------------------------------------------------
# 5. 그래프 저장
# -----------------------------------------------------------------------------
save_path = '../results/experiment_04_economic_performance.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[INFO] 그래프 이미지가 저장되었습니다: {save_path}")

# -----------------------------------------------------------------------------
# 6. 최종 요약 테이블 저장
# -----------------------------------------------------------------------------
summary_table = pd.DataFrame()
for strat_key, strat_label in strategies.items():
    for metric_suffix, metric_name in metrics.items():
        col = f'{strat_key}{metric_suffix}'
        if col in df.columns:
            summary_table.loc[strat_label, metric_name] = df[col].mean()

print("\n[Final Summary Table]")
print(summary_table.round(4))
