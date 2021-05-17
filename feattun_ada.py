import re

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(train[['Age']])
imp_mode = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mode.fit(test[['Fare']])
train['Age'] = imp.transform(np.array(train['Age']).reshape(-1, 1))
test['Age'] = imp.transform(np.array(test['Age']).reshape(-1, 1))
test['Fare'] = imp_mode.transform(np.array(test['Fare']).reshape(-1, 1))

le = LabelEncoder()
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)
test['Embarked'] = le.fit_transform(test['Embarked'])
embark_mappings = {index: label for index, label in enumerate(le.classes_)}

train['Sex'] = le.fit_transform(train['Sex'])
train_sex_mappings = {index: label for index, label in enumerate(le.classes_)}
test['Sex'] = le.fit_transform(test['Sex'])
test_sex_mappings = {index: label for index, label in enumerate(le.classes_)}

train_test = [train, test]
for data in train_test:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


for data in train_test:
    data['Title'] = data['Name'].apply(get_title)

for data in train_test:
    data['Title'] = data['Title'].replace(['Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                           'Rev', 'Sir', 'Jonkeer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Age_bin'] = pd.cut(data['Age'], bins=[0, 12, 20, 50, 70, 120],
                             labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Elder'])
    data['Fare_bin'] = pd.cut(data['Fare'], bins=[0, 7.91, 14.45, 31, 120],
                              labels=['Min_fare', 'Median_fare', 'Mean_fare', 'Max_fare'])

features = ['Pclass', 'Sex', 'FamilySize', 'Age_bin', 'Fare_bin', 'Embarked']
parameters = {'n_estimators': [5, 10, 15, 20, 25], 'max_depth': [3, 5, 7, 9, 11, 13]}
X = pd.get_dummies(train[features])
y = train['Survived']
X_test = pd.get_dummies(test[features])

clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X, y)

survived = clf.predict(X_test)
survivor_dict = {'PassengerId': list(test['PassengerId']), 'Survived': list(survived)}
survivor_df = pd.DataFrame(survivor_dict, columns=['PassengerId', 'Survived'])
