import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(train[['Age']])
train['Age'] = imp.transform(np.array(train['Age']).reshape(-1, 1))
test['Age'] = imp.transform(np.array(test['Age']).reshape(-1, 1))

le = LabelEncoder()
train['Embarked'].fillna('S', inplace=True)
train['Embarked'] = le.fit_transform(train['Embarked'])
embark_mappings = {index: label for index, label in enumerate(le.classes_)}

test['Fare'].fillna(test['Fare'].mode(), inplace=True)

train['Sex'] = le.fit_transform(train['Sex'])
train_sex_mappings = {index: label for index, label in enumerate(le.classes_)}
test['Sex'] = le.fit_transform(test['Sex'])
test_sex_mappings = {index: label for index, label in enumerate(le.classes_)}
