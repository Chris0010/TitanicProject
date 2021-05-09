import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

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
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X = sc.fit_transform(X)
# X_test = sc.transform(X_test)
classifier = LogisticRegression()
rfe = RFE(classifier, n_features_to_select=3)
rfe.fit(X, y)
print(rfe.score(X, y))
'''
# knn was the best estimator without any type of optimization
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)
'''
