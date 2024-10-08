import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
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

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Gradient Boosting 회귀 모델 생성 및 훈련
model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_gb.fit(X_train, Y_train)

# 모델 예측
y_pred_gb = model_gb.predict(X_test)

# 성능 평가
mae_gb = mean_absolute_error(Y_test, y_pred_gb)
mse_gb = mean_squared_error(Y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)

print(f'Gradient Boosting MAE: {mae_gb:.2f}')
print(f'Gradient Boosting MSE: {mse_gb:.2f}')
print(f'Gradient Boosting RMSE: {rmse_gb:.2f}')

# Density Plot of Actual vs Predicted Life Expectation 시각화
plt.figure(figsize=(10, 6))
data = pd.read_csv("./results/y_pred.csv")
re_ypred = data['Life expectation_pred']
re_ytest = data['Life expectation_test']
plt.scatter(re_ytest, re_ypred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # 45도 기준선
plt.xlabel('Actual Life Expectation')
plt.ylabel('Predicted Life Expectation')
plt.title('Actual vs Predicted Life Expectation (Gradient Boosting)')
plt.grid(True)
plt.savefig('actual_vs_predicted_life_expectation(gb).png', dpi=300, bbox_inches='tight')
plt.close()
