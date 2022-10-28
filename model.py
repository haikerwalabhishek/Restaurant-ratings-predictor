import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('zomato_preprocessed.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())
x=df.drop('rate',axis=1)
y=df['rate']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)

#preparing extra tree
from sklearn.ensemble import ExtraTreesRegressor
ET=ExtraTreesRegressor(n_estimators=120)
ET.fit(x_train,y_train)
y_pred=ET.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


import pickle
pickle.dump(ET,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_pred)