from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
# 데이터 로드
data = pd.read_csv('./data/4.power_transform.csv')

# 독립 변수(X)와 종속 변수(Y) 설정
X = data[['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO2', 'DBDS', 'Interfacial V', 'Dielectric rigidity',
          'Water content', 'Health index']]
Y = data['Life expectation']

# K-Fold 교차 검증 설정
kf = KFold(n_splits=100, shuffle=True, random_state=42)

models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf'),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

for model_name, model in models.items():
    mae_scores = []
    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # 모델 훈련
        model.fit(X_train, Y_train)

        # 모델 예측
        y_pred = model.predict(X_test)

        # 성능 평가
        mae_scores.append(mean_absolute_error(Y_test, y_pred))
        mse_scores.append(mean_squared_error(Y_test, y_pred))
        r2_scores.append(r2_score(Y_test, y_pred))

    print(f'{model_name} MAE: {np.mean(mae_scores):.2f}')
    print(f'{model_name} MSE: {np.mean(mse_scores):.2f}')
    print(f'{model_name} R^2: {np.mean(r2_scores):.2f}\n')