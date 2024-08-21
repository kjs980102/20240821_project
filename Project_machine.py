#가설 변압기 수명 예측 프로그램

#– 건설 현장에서 사용하는 가설 변압기는 다양한 이유로 고장 발생
#– 가장 흔한 원인으로는 번개, 과부하, 마모 및 부식, 전력 서지, 그리고 습기등이 원인
#– 변압기가 고장나면 대형 화재 및 인명피해 발생 가능
#– 변압기 고장 예방을 위해 다양한 데이터를 기반으로 변압기의 수명을 예측
#– 471개의 데이터 및 16개의 속성
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/4.power_transform.csv')
# 가설 변압기와 수소와의 상관관계
X1 = data['Hydrogen']
Y1 = data['Life expectation']
plt.figure(figsize=(10,6) )
correlation_coefficient = pearsonr(X1, Y1)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X1, Y1, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 산소와의 상관관계
X2 = data['Oxigen']
Y2 = data['Life expectation']
plt.figure(figsize=(10,6) )
plt.scatter(X2, Y2, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
correlation_coefficient = pearsonr(X2, Y2)
print(f'상관계수: {correlation_coefficient}')
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 질소와의 상관관계
X3 = data['Nitrogen']
Y3 = data['Life expectation']
plt.figure(figsize=(10,6) )
plt.scatter(X3, Y3, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
correlation_coefficient = pearsonr(X3, Y3)
print(f'상관계수: {correlation_coefficient}')
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 메탄과의 상관관계
X4 = data['Methane']
Y4 = data['Life expectation']
plt.figure(figsize=(10,6) )
plt.scatter(X4, Y4, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
correlation_coefficient = pearsonr(X4, Y4)
print(f'상관계수: {correlation_coefficient}')
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 일산화탄소와의 상관관계
X5 = data['CO']
Y5 = data['Life expectation']
plt.figure(figsize=(10,6) )
plt.scatter(X5, Y5, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
correlation_coefficient = pearsonr(X5, Y5)
print(f'상관계수: {correlation_coefficient}')
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 이산화탄소와의 상관관계
X6 = data['CO2']
Y6 = data['Life expectation']
plt.figure(figsize=(10,6) )
correlation_coefficient = pearsonr(X6, Y6)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X6, Y6, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 에틸렌과의 상관관계
X7 = data['Ethylene']
Y7 = data['Life expectation']
plt.figure(figsize=(10,6) )
correlation_coefficient = pearsonr(X7, Y7)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X7, Y7, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 에탄과의 상관관계
X8 = data['Ethane']
Y8 = data['Life expectation']
plt.figure(figsize=(10,6) )
correlation_coefficient = pearsonr(X8, Y8)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X8, Y8, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 아세틸렌과의 상관관계
X9 = data['Acethylene']
Y9 = data['Life expectation']
plt.figure(figsize=(10,6) )
correlation_coefficient = pearsonr(X9, Y9)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X9, Y9, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 디벤질 디설파이드와의 상관관계
X10= data['DBDS']
Y10 = data['Life expectation']
plt.figure(figsize=(10,6) )
correlation_coefficient = pearsonr(X10, Y10)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X9, Y9, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
plt.title('d')
plt.xlabel('ppm')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 전력계수와의 상관관계
X10 = data['Power factor']
Y10 = data['Life expectation']
plt.figure(figsize=(10,6) )
plt.scatter(X10, Y10, color='blue', label='Life expectation', marker='o' ,s=30, alpha=0.5)
plt.title('d')
plt.xlabel('')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 계면전압과의 상관관계
X11 = data['Interfacial V']
Y11 = data['Life expectation']
plt.figure(figsize=(10,6))
correlation_coefficient = pearsonr(X11, Y11)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X11, Y11, color='blue', label='Life expectation', marker='o', s=30, alpha=0.5)
plt.title('d')
plt.xlabel('')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 절연강도와의 상관관계
X12 = data['Dielectric rigidity']
Y12 = data['Life expectation']
plt.figure(figsize=(10,6))
correlation_coefficient = pearsonr(X12, Y12)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X12, Y12, color='blue', label='Life expectation', marker='o', s=30, alpha=0.5)
plt.title('d')
plt.xlabel('')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 수분함량과의 상관관계
X13 = data['Water content']
Y13 = data['Life expectation']
plt.figure(figsize=(10,6))
correlation_coefficient = pearsonr(X13, Y13)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X13, Y13, color='blue', label='Life expectation', marker='o', s=30, alpha=0.5)
plt.title('d')
plt.xlabel('')
plt.ylabel('Years')
plt.legend()
plt.show()
#가설 변압기와 변압기 상태와의 상관관계
X14 = data['Health index']
Y14 = data['Life expectation']
plt.figure(figsize=(10,6))
correlation_coefficient = pearsonr(X14, Y14)
print(f'상관계수: {correlation_coefficient}')
plt.scatter(X14, Y14, color='blue', label='Life expectation', marker='o', s=30, alpha=0.5)
plt.title('d')
plt.xlabel('')
plt.ylabel('Years')
plt.legend()
plt.show()



# array = data.values
# X = array[:, 0:14]
# Y = array[:, 14]
# print(X)
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# model = LinearRegression()
# model.fit(X_train, Y_train)
# model.predict(X_test)
# y_pred = model.predict(X_test)
#
# print(model.coef_, model)
# plt.scatter(range(len(X_test[:15])), Y_test[:15], color='blue')
# plt.scatter(range(len(X_test[:15])), y_pred[:15], color='red', marker='x')
# plt.xlabel("material")
# plt.ylabel("Life expectation")
# plt.show()
