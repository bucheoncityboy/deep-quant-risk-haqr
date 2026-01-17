# src/models.py

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Lambda, Multiply, 
    GlobalAveragePooling1D, Reshape, Softmax, Add, Activation
)
from tensorflow.keras.models import Model

# [수정] 예측할 분위수 (전역 설정, 90% 구간)
QUANTILES = [0.05, 0.5, 0.95] 

def pinball_loss(y_true, y_pred):
    """[핵심 기여 3] 핀볼 손실 함수 (Quantile Regression Loss)"""
    y_true_replicated = K.repeat_elements(y_true, len(QUANTILES), axis=1)
    q = tf.constant(np.array(QUANTILES, dtype=np.float32).reshape(1, -1))
    e = y_true_replicated - y_pred
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

def build_haqr_model(num_trend_features, num_market_features, h_dim=8):
    """
    [핵심 기여 1] HAQR 모델 구축 함수
    Returns:
        haqr_model (Trainable), group_att_model (XAI), factor_att_model (XAI)
    """
    
    # --- 내부 함수: 1단계 팩터-레벨 어텐션 인코더 ---
    def factor_attention_encoder(input_tensor, num_features, h_dim, group_name):
        reshaped = Reshape((num_features, 1), name=f'{group_name}_factor_reshape')(input_tensor)
        factor_embeddings = Dense(h_dim, activation='relu', name=f'{group_name}_factor_embed')(reshaped)
        factor_scores = Dense(1, activation='linear', name=f'{group_name}_factor_score')(factor_embeddings)
        factor_weights = Softmax(axis=1, name=f'{group_name}_factor_weights')(factor_scores)
        weighted_factors = Multiply(name=f'{group_name}_factor_weighted')([factor_embeddings, factor_weights])
        group_vector = GlobalAveragePooling1D(name=f'{group_name}_factor_context')(weighted_factors)
        return group_vector, factor_weights
    
    # 1. 입력 레이어
    input_trend = Input(shape=(num_trend_features,), name='trend_input', dtype=tf.float32)
    input_market = Input(shape=(num_market_features,), name='market_input', dtype=tf.float32)

    # 2. 1단계: 팩터-레벨 어텐션 (HAN Body)
    h_trend, factor_weights_trend = factor_attention_encoder(input_trend, num_trend_features, h_dim, 'trend')
    h_market, factor_weights_market = factor_attention_encoder(input_market, num_market_features, h_dim, 'market')

    # 3. 2단계: 그룹-레벨 어텐션 (HAN Body)
    stacked_groups = Lambda(lambda x: tf.stack(x, axis=1), name='stacked_groups')([h_trend, h_market])
    attention_scores = Dense(1, activation='linear', name='attention_scorer')(stacked_groups) 
    attention_weights_group = Softmax(axis=1, name='attention_weights_group')(attention_scores)
    weighted_groups = Multiply(name='weighted_groups')([stacked_groups, attention_weights_group])
    context_vector = GlobalAveragePooling1D(name='context_vector')(weighted_groups) 

    # 4. 3단계: Non-Crossing 퀸타일 헤드 (Output) - [수정] 0.05, 0.5, 0.95
    base_output = Dense(h_dim, activation='relu', name='base_head')(context_vector)
    
    # Q(0.05)
    output_q05 = Dense(1, name='q_0.05')(base_output)
    
    # Q(0.50) = Q(0.05) + Delta1
    output_q50_delta = Dense(1, activation='softplus', name='q_0.50_delta')(base_output)
    output_q50 = Add(name='q_0.50')([output_q05, output_q50_delta])
    
    # Q(0.95) = Q(0.50) + Delta2
    output_q95_delta = Dense(1, activation='softplus', name='q_0.95_delta')(base_output)
    output_q95 = Add(name='q_0.95')([output_q50, output_q95_delta])
    
    output_main = Concatenate(name='quantile_output')([output_q05, output_q50, output_q95])
    
    # 모델 정의
    haqr_model = Model(
        inputs={'trend_input': input_trend, 'market_input': input_market}, 
        outputs=output_main, name="HAQR_Model"
    )
    attention_model_group = Model(
        inputs={'trend_input': input_trend, 'market_input': input_market},
        outputs=attention_weights_group, name="HAQR_Group_Attention_Extractor"
    )
    attention_model_factor = Model(
        inputs={'trend_input': input_trend, 'market_input': input_market},
        outputs={'trend_factors': factor_weights_trend, 'market_factors': factor_weights_market},
        name="HAQR_Factor_Attention_Extractor"
    )
    
    return haqr_model, attention_model_group, attention_model_factor
