import pandas as pd
df=pd.read_csv("C:/Users/ACER/PycharmProjects/DataScience/IRIS.csv")

x=df.drop('species',axis=1)
y=df['species']
print(x)
print(y)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featuresScores = pd.concat([dfcolumns,dfscores], axis=1)
featuresScores.columns = ['Specs', 'Score']
print(featuresScores)

print(df.isnull().sum())
df['petal_length'].fillna((df['petal_length'].mean()), inplace=True)
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['species']=le.fit_transform(df['species'])
print(df)

from imblearn.over_sampling import SMOTE
sms = SMOTE(random_state=0)
x,y = sms.fit_resample(x,y)

from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['sepal_length'])
plt.show()

print(df['sepal_length'])
Q1=df['sepal_length'].quantile(0.25)
Q3=df['sepal_length'].quantile(0.75)
IQR=Q3-Q1
print("IQR:",IQR)
upper=Q3+1.5*IQR
lower=Q1-1.5*IQR
print(upper)
print(lower)
out1=df[df['sepal_length']<lower].values
out2=df[df['sepal_length']>upper].values
df['sepal_length'].replace(out1,lower,inplace=True)
df['sepal_length'].replace(out2,upper,inplace=True)

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression()
pca=PCA(n_components=2)

x=df.drop('species',axis=1)
y=df['species']

pca.fit(x)
x=pca.transform(x)

print(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0,test_size=0.3)
logr.fit(xtrain,ytrain)
ypred=logr.predict(xtest)
print(accuracy_score(ytest,ypred))
