# 가설 변압기 수명 예측 프로그램

1. 데이터 로드(4.power_transform)

2. 상관관계 분석
- Life expectation과 다른 변수들 간의 상관계수 행렬 계산
- 더 나은 시각화를 위해 Life expectation과의 상관계수만 포함된 서브셋 생성
- 상관
3. 교차 검증 및 적합한 머신러닝 선정
-머신러닝 모델 정확도 비교.py
-
4. Random forest, Gradient Boosting 을 사용하여 모델 생성 및 훈련

5. 예측 모델 시각화