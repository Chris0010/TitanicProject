import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(train[['Age']])
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mode.fit(test[['Fare']])
train['Age'] = imp.transform(np.array(train['Age']).reshape(-1, 1))
test['Age'] = imp.transform(np.array(test['Age']).reshape(-1, 1))
test['Fare'] = imp_mode.transform(np.array(test['Fare']).reshape(-1, 1))

le = LabelEncoder()
train['Embarked'].fillna('S', inplace=True)
train['Embarked'] = le.fit_transform(train['Embarked'])
embark_mappings = {index: label for index, label in enumerate(le.classes_)}

train['Sex'] = le.fit_transform(train['Sex'])
train_sex_mappings = {index: label for index, label in enumerate(le.classes_)}
test['Sex'] = le.fit_transform(test['Sex'])
test_sex_mappings = {index: label for index, label in enumerate(le.classes_)}

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Age', 'Fare', 'Embarked']
y = train['Survived']
X = pd.get_dummies(train[features])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# X_test = pd.get_dummies(test[features])
# y_test = test['Survived']

lr = LogisticRegression()
lr.fit(X, y)
y_pred = lr.predict(X_test)

'''
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
'''
'''
rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
'''
'''
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
'''
'''
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
'''
