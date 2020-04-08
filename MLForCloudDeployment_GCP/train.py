import pandas as pd
import matplotlib.pyplot as plt
import pickle

df= pd.read_csv('Admission_Prediction.csv')

df.head()

df.info()
df['GRE Score'].fillna(df['GRE Score'].mode()[0],inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mode()[0],inplace=True)
df['University Rating'].fillna(df['University Rating'].mean(),inplace=True)

x=df.drop(['Chance of Admit','Serial No.'],axis=1)
y=df['Chance of Admit']

plt.scatter(df['GRE Score'],y)
plt.show()
plt.scatter(df['TOEFL Score'],y)
plt.show()
plt.scatter(df['CGPA'],y)
plt.show()

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33, random_state=100)
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(train_x, train_y)
from sklearn.metrics import r2_score
score= r2_score(reg.predict(test_x),test_y)

filename = 'finalized_model.pickle'
pickle.dump(reg, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
prediction=loaded_model.predict(([[320,120,5,5,5,10,1]]))
print(prediction[0])
