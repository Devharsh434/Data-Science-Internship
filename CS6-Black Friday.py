import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# import matplotlib_inline

rf = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier(random_state=1)
df_train = pd.read_csv("C:/Users/admin/PycharmProjects/pythonProject/Black Friday/train.csv")
df_test = pd.read_csv("C:/Users/admin/PycharmProjects/pythonProject/Black Friday/test.csv")
print(df_train.keys())
# print(df_test.keys())
#
# print(df_train.isnull().sum())
# print(df_test.isnull().sum())

df = df_train._append(df_test)

df.drop(['User_ID'], axis=1, inplace=True)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Age'] = le.fit_transform(df['Age'])
df['City_Category'] = le.fit_transform(df['City_Category'])
print(df['City_Category'])
print(df['Age'])

print(df.isnull().sum())

df['Product_Category_2'].fillna((df['Product_Category_2'].mean()), inplace=True)
df['Product_Category_3'].fillna((df['Product_Category_3'].mean()), inplace=True)

print(df.isnull().sum())

print(df['Stay_In_Current_City_Years'].unique())
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace('+', '')
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)
print(df.info())

# sns.barplot(x='Age', y='Purchase', hue='Gender', data=df)
# plt.show()
# sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)
# plt.show()
# sns.barplot(x='Product_Category_1',y='Purchase',hue='Gender',data=df)
# plt.show()
# sns.barplot(x='Product_Category_2',y='Purchase',hue='Gender',data=df)
# plt.show()
# sns.barplot(x='Product_Category_3',y='Purchase',hue='Gender',data=df)
# plt.show()

df_test=df[df['Purchase'].isnull()]
df_train=df[~df['Purchase'].isnull()]

X=df_train.drop('Purchase',axis=1)
y=df_train['Purchase']

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

lr=GradientBoostingRegressor()
X_train,X_test,Y_train,Y_test = train_test_split(X,y,random_state=0,test_size=0.3)
X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
print(r2_score(Y_test,y_pred))


