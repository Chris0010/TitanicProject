import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

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

kfold = model_selection.KFold(n_splits=10)
estimators = []
logreg = LogisticRegression(max_iter=10000)
rfe = RFE(logreg, n_features_to_select=1)
rfe.fit(X, y)
estimators.append(('logistic', rfe))
tree = RandomForestClassifier(max_features='log2')
param_grid = {'n_estimators': [800], 'max_features': ['auto', 'sqrt', 'log2']}
clf = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5)
estimators.append(('cart', clf))
svc = SVC()
estimators.append(('svm', svc))
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)
print(results.mean())
