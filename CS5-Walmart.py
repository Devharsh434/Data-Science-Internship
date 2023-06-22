import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

features = pd.read_csv("C:/Users/admin/PycharmProjects/pythonProject/Walmart/features.csv/features.csv")
train = pd.read_csv("C:/Users/admin/PycharmProjects/pythonProject/Walmart/train.csv/train.csv")
test1 = pd.read_csv("C:/Users/admin/PycharmProjects/pythonProject/Walmart/test.csv/test.csv")
stores = pd.read_csv("C:/Users/admin/PycharmProjects/pythonProject/Walmart/stores.csv")

df1 = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='inner')
df = df1.merge(stores, on=['Store'], how='inner')
print(df.head())

print(df.isnull().sum())

X = df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1)
print(X.isnull().sum())

X['Date'] = pd.to_datetime(X['Date'])
X.set_index(keys = "Date", inplace = True)

X['Temperature'].hist()

print(df.shape)
out = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
for i in out:
    # sns.boxplot(df[i])
    # plt.show()
    # print(X[i])
    Q1 = X[i].quantile(0.25)
    Q3 = X[i].quantile(0.75)
    IQR = Q3 - Q1
    # print(IQR)
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    # print(upper)
    # print(lower)
    out1 = X[X[i] < lower].values
    out2 = X[X[i] > upper].values
    X[i].replace(out1, lower, inplace=True)
    X[i].replace(out2, upper, inplace=True)
    # sns.boxplot(X[i])
    # plt.show()

df_test = test1.merge(features, on = ['Store', 'Date', 'IsHoliday'], how = 'inner')
test = df_test.merge(stores, on = ['Store'], how = 'inner')

test = test.drop(["MarkDown1", "MarkDown2","MarkDown3","MarkDown4", "MarkDown5"], axis=1)
print(test.head())

print(test.isnull().sum())
test['CPI'] = test['CPI'].fillna(test['CPI'].mean())
test['Unemployment'] = test['Unemployment'].fillna(test['Unemployment'].mean())
print(test.isnull().sum())

out = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
for i in out:
    # sns.boxplot(df[i])
    # plt.show()
    # print(X[i])
    Q1 = test[i].quantile(0.25)
    Q3 = test[i].quantile(0.75)
    IQR = Q3 - Q1
    # print(IQR)
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    # print(upper)
    # print(lower)
    out1 = test[test[i] < lower].values
    out2 = test[test[i] > upper].values
    test[i].replace(out1, lower, inplace=True)
    test[i].replace(out2, upper, inplace=True)
    # sns.boxplot(X[i])
    # plt.show()

test['Date'] = pd.to_datetime(test['Date'])
test.set_index(keys = 'Date', inplace = True)

le = LabelEncoder()
X['IsHoliday'] = le.fit_transform(X['IsHoliday'])
X['Type'] = le.fit_transform(X['Type'])
test['IsHoliday'] = le.fit_transform(test['IsHoliday'])
test['Type'] = le.fit_transform(test['Type'])

X.drop(['Size'], axis = 1, inplace = True)
corr = X.corr()

x = X.drop(['Weekly_Sales'], axis = 1)
y = X['Weekly_Sales']
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
dtmodel=DecisionTreeRegressor()

x_train,x_test,y_train,y_test= train_test_split(x, y, random_state=5,test_size=0.3)
print("x_train size:",x_train.shape)
print("x_test size:",x_test.shape)
print("y_train size:",y_train.shape)
print("y_test size:",y_test.shape)

from sklearn.linear_model import LinearRegression
lrmodel= LinearRegression()
lrmodel.fit(x_train,y_train)
prediction1=lrmodel.predict(x_test)
print(lrmodel.score(x_test,y_test))

from sklearn.linear_model import LinearRegression
lrmodel= LinearRegression()
lrmodel.fit(x_train,y_train)
prediction1=lrmodel.predict(x_test)
lrmodel.score(x_test,y_test)

mse = mean_squared_error(prediction1,y_test)
print("mean sqaure error:",mse)
from sklearn.metrics import r2_score
print(r2_score(prediction1,y_test))

user_input=np.array([[1,1,0,42.21,2.572,211.096,8.106,0]])
prediction=lrmodel.predict(user_input)
print(prediction)