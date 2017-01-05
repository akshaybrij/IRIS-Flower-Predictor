import numpy as np
import pandas as pd
from sklearn import preprocessing,cross_validation,neighbors,svm
import warnings

df=pd.read_csv('iris.txt')

df.replace('?',-9999,inplace=True)
#df.drop(['id'],1,inplace=True)
X=(df.drop(['class'],1))
print df.head()
Y=df['class']
def fx():
    warnings.warn("Deprecated",DeprecationWarning)
with warnings.catch_warnings() :
    warnings.simplefilter("always")
    fx()
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.3)
clf=svm.SVC()
clf.fit(X_train,Y_train)
acc=clf.score(X_test,Y_test)
print acc*100
ds=np.array([4.9,3.5,2.2,0.2])
print type(ds)
ds.reshape(-1,1)
s=clf.predict(ds)
print s
