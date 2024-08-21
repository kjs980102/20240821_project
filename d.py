import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#데이터가 쉼표로 구분되어 있지 않고 스페이스바로 구분시켜 놓았을 때는 deli_whitespace 사용
data = pd.read_csv('./data/3.housing.csv', delim_whitespace=True, names=header)
array = data.values
#독립변수 종속변수 나누기
X = array[:, 0:13]
Y = array[:, 13]
Y = Y.reshape(-1, 1)
print(Y)
#학습 데이터 / 테스트 데이터
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#선형회귀 모델 = LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
model.predict(X_test)
y_pred = model.predict(X_test)

print(model.coef_, model)
plt.scatter(range(len(X_test[:15])), Y_test[:15], color='blue')
plt.scatter(range(len(X_test[:15])), y_pred[:15], color='red', marker='x')
plt.xlabel("Index")
plt.ylabel("MEDV ($1000)")
plt.show()
mse=mean_squared_error(Y_test, y_pred)