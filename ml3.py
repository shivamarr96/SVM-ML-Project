#Task 3
#The columns with small values are dropped and normalisation is done using Min-Max Scaling 

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

ac_score=[]
ac_score1=[]
ac_score2=[]
ac_score3=[]

data = pd.read_csv("aman_ml_3_authors.csv")
y = data.author_id

data1 = pd.read_csv("aman_ml_authors_10.csv")
y1 = data1.author_id

data2 = pd.read_csv("aman_ml_authors_15.csv")
y2 = data2.author_id

data3 = pd.read_csv("aman_ml_authors_20.csv")
y3 = data3.author_id

#removing columns with small values

X = data.drop('author_id', axis = 1)
X = X.drop('cconj',axis=1)
X = X.drop('sym',axis=1)
X = X.drop('intj',axis=1)
X = X.drop('punct',axis=1)

X1 = data1.drop('author_id', axis = 1)
X1 = X1.drop('cconj',axis=1)
X1 = X1.drop('sym',axis=1)
X1 = X1.drop('intj',axis=1)
X1 = X1.drop('punct',axis=1)

X2 = data2.drop('author_id', axis = 1)
X2 = X2.drop('cconj',axis=1)
X2 = X2.drop('sym',axis=1)
X2 = X2.drop('intj',axis=1)
X2 = X2.drop('punct',axis=1)

X3 = data3.drop('author_id', axis = 1)
X3 = X3.drop('cconj',axis=1)
X3 = X3.drop('sym',axis=1)
X3 = X3.drop('intj',axis=1)
X3 = X3.drop('punct',axis=1)

#before normalising
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 123)
X1_train, X1_test, y1_train, y1_test= train_test_split(X1, y1, test_size = .3, random_state = 123)
X2_train, X2_test, y2_train, y2_test= train_test_split(X2, y2, test_size = .3, random_state = 123)
X3_train, X3_test, y3_train, y3_test= train_test_split(X3, y3, test_size = .3, random_state = 123)

clf = SVC(kernel = 'linear', verbose = False, C = 1)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

clf.fit(X1_train,y1_train)
predictions1 = clf.predict(X1_test)

clf.fit(X2_train,y2_train)
predictions2 = clf.predict(X2_test)

clf.fit(X3_train,y3_train)
predictions3 = clf.predict(X3_test)

ac_score.append(accuracy_score(y_test, predictions))
ac_score1.append(accuracy_score(y1_test, predictions1))
ac_score2.append(accuracy_score(y2_test, predictions2))
ac_score3.append(accuracy_score(y3_test, predictions3))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 123)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = .3, random_state = 123)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = .3, random_state = 123)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = .3, random_state = 123)



# Standardization
x_np = np.asarray(X)
z_scores_np = (x_np - x_np.mean()) / x_np.std()

x_np1 = np.asarray(X1)
z_scores_np1 = (x_np1 - x_np1.mean()) / x_np1.std()

x_np2 = np.asarray(X2)
z_scores_np2 = (x_np2 - x_np2.mean()) / x_np2.std()

x_np3 = np.asarray(X3)
z_scores_np3 = (x_np3 - x_np3.mean()) / x_np3.std()

# Min-Max scaling
np_minmax = (x_np - x_np.min()) / (x_np.max() - x_np.min())
X_train,X_test,y_train,y_test=train_test_split(np_minmax,y,test_size=.3,random_state=123)

np_minmax1 = (x_np1 - x_np1.min()) / (x_np1.max() - x_np1.min())
X1_train,X1_test,y1_train,y1_test=train_test_split(np_minmax1,y1,test_size=.3,random_state=123)

np_minmax2 = (x_np2 - x_np2.min()) / (x_np2.max() - x_np2.min())
X2_train,X2_test,y2_train,y2_test=train_test_split(np_minmax2,y2,test_size=.3,random_state=123)

np_minmax3 = (x_np3 - x_np3.min()) / (x_np3.max() - x_np3.min())
X3_train,X3_test,y3_train,y3_test=train_test_split(np_minmax3,y3,test_size=.3,random_state=123)

clf = SVC(kernel='linear', verbose= False, C= 1)

clf.fit(X_train,y_train)
predictions=clf.predict(X_test)

clf.fit(X1_train,y1_train)
predictions1=clf.predict(X1_test)

clf.fit(X2_train,y2_train)
predictions2=clf.predict(X2_test)

clf.fit(X3_train,y3_train)
predictions3=clf.predict(X3_test)

X_train,X_test,y_train,y_test=train_test_split(np_minmax,y,test_size=.3,random_state=123)
ac_score.append(accuracy_score(y_test,predictions))

X1_train,X1_test,y1_train,y1_test=train_test_split(np_minmax1,y1,test_size=.3,random_state=123)
ac_score1.append(accuracy_score(y1_test,predictions1))

X2_train,X2_test,y2_train,y2_test=train_test_split(np_minmax2,y2,test_size=.3,random_state=123)
ac_score2.append(accuracy_score(y2_test,predictions2))

X3_train,X3_test,y3_train,y3_test=train_test_split(np_minmax3,y3,test_size=.3,random_state=123)
ac_score3.append(accuracy_score(y3_test,predictions3))
print "No. of Authors  Before Normalising  After Normalising"
print "3 authors",ac_score
print "10 authors",ac_score1
print "15 authors",ac_score2
print "20 authors",ac_score3