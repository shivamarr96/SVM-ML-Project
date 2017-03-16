#Task 2
#The SVC arguments are varied and the results are recorded into a list 

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import gmtime, strftime
import pandas as pd

print strftime("%Y-%m-%d %H:%M:%S", gmtime())
ac_score  =[]
ac_score1 =[]
ac_score2 =[]
ac_score3 =[]
ac_score.append('3 authors ')
ac_score1.append('10 authors ')
ac_score2.append('15 authors ')
ac_score3.append('20 authors ')

verbose_type = ['linear','linear with C=.0000000001','poly','rbf','rbf with C=.0000000001','sigmoid']

data  = pd.read_csv("aman_ml_3_authors.csv")
data1 = pd.read_csv("aman_ml_authors_10.csv")
data2 = pd.read_csv("aman_ml_authors_15.csv")
data3 = pd.read_csv("aman_ml_authors_20.csv")

y  = data.author_id
y1 = data1.author_id
y2 = data2.author_id
y3 = data3.author_id

X = data.drop('author_id', axis = 1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

X1 = data1.drop('author_id', axis = 1)
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=.3)

X2 = data2.drop('author_id', axis = 1)
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=.3)

X3 = data3.drop('author_id', axis = 1)
X3_train,X3_test,y3_train,y3_test=train_test_split(X3,y3,test_size=.3)

#linear with C=1

clf = SVC(kernel='linear', verbose= False, C= 1)

clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
ac_score.append(accuracy_score(y_test,predictions))

clf.fit(X1_train,y1_train)
predictions=clf.predict(X1_test)
ac_score1.append(accuracy_score(y1_test,predictions))

clf.fit(X2_train,y2_train)
predictions=clf.predict(X2_test)
ac_score2.append(accuracy_score(y2_test,predictions))

clf.fit(X3_train,y3_train)
predictions=clf.predict(X3_test)
ac_score3.append(accuracy_score(y3_test,predictions))

#for kernel=linear and c=.0000000001
#for very small values of c we get misclassified examples because the c parameter tells how much 
#we want to avoid misclassifying each training example


clf = SVC(kernel='linear', verbose= False, C= .0000000001)

clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
ac_score.append(accuracy_score(y_test,predictions))

clf.fit(X1_train,y1_train)
predictions=clf.predict(X1_test)
ac_score1.append(accuracy_score(y1_test,predictions))

clf.fit(X2_train,y2_train)
predictions=clf.predict(X2_test)
ac_score2.append(accuracy_score(y2_test,predictions))

clf.fit(X3_train,y3_train)
predictions=clf.predict(X3_test)
ac_score3.append(accuracy_score(y3_test,predictions))


#for kernel=poly
clf = SVC(kernel='poly', verbose= False, C= 1)

clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
ac_score.append(accuracy_score(y_test,predictions))

clf.fit(X1_train,y1_train)
predictions=clf.predict(X1_test)
ac_score1.append(accuracy_score(y1_test,predictions))

clf.fit(X2_train,y2_train)
predictions=clf.predict(X2_test)
ac_score2.append(accuracy_score(y2_test,predictions))

clf.fit(X3_train,y3_train)
predictions=clf.predict(X3_test)
ac_score3.append(accuracy_score(y3_test,predictions))
#for kernel=rbf and C=.0000000001
clf = SVC(kernel='rbf', verbose= False, C= .0000000001)
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
ac_score.append(accuracy_score(y_test,predictions))

clf.fit(X1_train,y1_train)
predictions=clf.predict(X1_test)
ac_score1.append(accuracy_score(y1_test,predictions))

clf.fit(X2_train,y2_train)
predictions=clf.predict(X2_test)
ac_score2.append(accuracy_score(y2_test,predictions))

clf.fit(X3_train,y3_train)
predictions=clf.predict(X3_test)
ac_score3.append(accuracy_score(y3_test,predictions))


print strftime("%Y-%m-%d %H:%M:%S", gmtime())

from tabulate import tabulate
print tabulate([ac_score,ac_score1,ac_score2,ac_score3],headers=['author numbers','linear','linear with C=.0000000001','poly','rbf','rbf with C=.0000000001'])