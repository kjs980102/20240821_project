import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


# 데이터 로드
data = pd.read_csv('./data/4.power_transform.csv')

# Life expectation과 다른 변수들 간의 상관계수 행렬 계산
correlation_with_life_expectation = data[['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content', 'Health index', 'Life expectation']].corr()

# Life expectation과의 상관계수만 포함된 서브셋 생성
life_expectation_corr = correlation_with_life_expectation[['Life expectation']]
life_expectation_corr = life_expectation_corr.drop('Life expectation')

# 상관계수의 부호를 포함하여 정렬 (내림차순)
sorted_corr = life_expectation_corr.sort_values(by='Life expectation', ascending=False)

# 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(sorted_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=0.5)
plt.title('Correlation with Life Expectation (Sorted by Correlation)')
plt.show()
# 독립 변수(X)와 종속 변수(Y) 설정
X = data[['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO2', 'DBDS', 'Interfacial V', 'Dielectric rigidity',
          'Water content', 'Health index']]
Y = data['Life expectation']

# K-Fold 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=42)

mae_scores = []
mse_scores = []
r2_scores = []

# K-Fold 교차 검증 수행
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Gradient Boosting 회귀 모델 생성 및 훈련
    model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_gb.fit(X_train, Y_train)

    # 모델 예측
    y_pred_gb = model_gb.predict(X_test)

    # 성능 평가
    mae_gb = mean_absolute_error(Y_test, y_pred_gb)
    mse_gb = mean_squared_error(Y_test, y_pred_gb)
    r2_gb = r2_score(Y_test, y_pred_gb)

    print(f'Gradient Boosting MAE: {mae_gb:.2f}')
    print(f'Gradient Boosting MSE: {mse_gb:.2f}')
    print(f'Gradient Boosting R^2: {r2_gb:.2f}')

    plt.figure(figsize=(10, 6))
    sns.kdeplot(Y_test, label='Actual Life Expectation', shade=True, color='blue')
    sns.kdeplot(y_pred_gb, label='Predicted Life Expectation', shade=True, color='orange')
    plt.xlabel('Life Expectation')
    plt.title('Density Plot of Actual vs Predicted Life Expectation')
    plt.legend()
    plt.show()

# # 성능 지표 평균 출력
# print(f'Average Mean Absolute Error (MAE): {np.mean(mae_scores):.2f}')
# print(f'Average Mean Squared Error (MSE): {np.mean(mse_scores):.2f}')
# print(f'Average R^2 Score: {np.mean(r2_scores):.2f}')
#
# # 각 폴드에서의 성능 지표를 시각화
# fig, ax = plt.subplots(3, 1, figsize=(12, 12))

# # MAE 시각화
# ax[0].plot(range(1, 11), mae_scores, marker='o', linestyle='-', color='b', label='MAE')
# ax[0].set_title('Mean Absolute Error (MAE) per Fold')
# ax[0].set_xlabel('Fold')
# ax[0].set_ylabel('MAE')
# ax[0].legend()
#
# # MSE 시각화
# ax[1].plot(range(1, 11), mse_scores, marker='o', linestyle='-', color='r', label='MSE')
# ax[1].set_title('Mean Squared Error (MSE) per Fold')
# ax[1].set_xlabel('Fold')
# ax[1].set_ylabel('MSE')
# ax[1].legend()
#
# # R² Score 시각화
# ax[2].plot(range(1, 11), r2_scores, marker='o', linestyle='-', color='g', label='R² Score')
# ax[2].set_title('R² Score per Fold')
# ax[2].set_xlabel('Fold')
# ax[2].set_ylabel('R² Score')
# ax[2].legend()
