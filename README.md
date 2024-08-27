# 가설 변압기 수명 예측 프로그램

1. 데이터 로드(4.power_transform)

2. 상관관계 분석
- Life expectation과 다른 변수들 간의 상관계수 행렬 계산
- 더 나은 시각화를 위해 Life expectation과의 상관계수만 포함된 서브셋 생성
  life_expectation_corr = correlation_with_life_expectation[['Life expectation']]
  life_expectation_corr = life_expectation_corr.drop('Life expectation')
- corr 함수를 이용하여 0.1 미만의 상관관계를 가진 데이터 제외(일산화탄소, 에틸렌, 에탄, 아세틸렌, 전력계수)
- hitmap 이용하여 상관관계 시각화
3. 교차 검증 및 적합한 머신러닝 선정
- 머신러닝 모델 정확도 비교.py 사용
- K-Fold 교차 검증 설정
- Decision Tree, Gradient Boosting, Random Forest
- mse,mae,r2를 이용하여 모델 정확도 평가
- Gradient Boosting, Random Forest 선정
4. Random forest, Gradient Boosting 을 사용하여 모델 생성 및 훈련
- 각각의 회귀 모델을 사용하여 훈련
- 모델 생성 및 예측
  y_pred_() = model_rf.predict(X_test)
- X = 종속변수: 수소, 산소, 질소, 메탄, 이산화탄소, DBDS, 계면전압, 절연강도, water content(함수량), Health index(건전도)
- Y = 독립변수: 가설 변압기 수명(Life expectation
- mae, mse, rmse를 이용하여 모델 성능 평가
  mae_rf = mean_absolute_error(Y_test, y_pred_rf)
  mse_rf = mean_squared_error(Y_test, y_pred_rf)
  rmse_rf = np.sqrt(mse_rf)

5. 예측 모델 시각화
- plot/45도 점선(실제 데이터 값)
- scatter/파란색 점(모델을 통한 예측값)
- 점선에 점이 가까울수록 오차 적음
- 최대 오차 약 2년