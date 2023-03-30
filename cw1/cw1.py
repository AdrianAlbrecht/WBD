import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#%matplotlib inline

dataset = pd.read_csv('cw1\Kruk_VLagun_156.csv')
print(dataset.shape)

print(dataset.describe())

dataset.plot(x='SS', y='SecchDisc', style='o')
plt.title('SS vs SecchDisc')
plt.xlabel('SS')
plt.ylabel('SecchDisc')
#plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.histplot(dataset['SecchDisc'])
#plt.show()

X = dataset['SS'].values.reshape(-1,1)
Y = dataset['SecchDisc'].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

regressor =  LinearRegression()
regressor.fit(X_train, Y_train)
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_pred.flatten()})

df1 = df.head(25)
df1.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5',color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()

plt.scatter(X_test, Y_test, color="gray")
plt.plot(X_test, y_pred, color="red", linewidth=2)
#plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
