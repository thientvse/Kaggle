import sweetviz
import pandas as pd
import random

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# my_report = sweetviz.compare([train, "Train"], [test, "Test"], "Survived")

# my_report.show_html("Report.html") # Not providing a filename will default to SWEETVIZ_REPORT.html


print(train.head())

# count empty
print(train.isna().sum())

#Look at all of the values in each column & get a count
# for val in train:
#    print(train[val].value_counts())
#    print()

test_df = test



drop_features = ['Name', 'PassengerId', 'Ticket','Cabin']

train = train.drop(drop_features, axis=1)
test = test.drop(drop_features, axis=1)

data = [train, test]


for dataset in data:
    dataset['Sex'] = dataset['Sex'].apply(lambda x:1 if x == 'female' else 0)


print(train.head())

# missing value age
for dataset in data:
   for i in range(0, 2):  # Iterating over 'Sex' 0 or 1
      for j in range(0, 3):  # Iterating over 'Pclass' 1, 2 or 3
         guess_df = dataset.loc[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()

         age_guess = guess_df.median()

         dataset.loc[
            (dataset['Age'].isnull()) & (dataset['Sex'] == i) & (dataset['Pclass'] == j + 1), ['Age']] = age_guess

   dataset['Age'] = dataset['Age'].astype(int)

print(train.head())

# wrangling
train['AgeBand'] = pd.cut(train['Age'], 5)
print(train[['AgeBand', 'Survived']].groupby(['AgeBand']).mean())


for dataset in data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age']) > 16 & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age']) > 32 & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age']) > 48 & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age']) > 64 & (dataset['Age'] <= 80), 'Age'] = 4

print(train.head())

train.drop(['AgeBand'], axis=1, inplace=True)


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

train['Fareband'] = pd.qcut(train['Fare'], 4)
train[['Fareband', 'Survived']].groupby(['Fareband']).mean()


for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


train.drop(['Fareband'], axis=1, inplace=True)
print(train.head())

# Embarked
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(train['Embarked'].dropna().mode()[0])

print(train[['Embarked', 'Survived']].groupby('Embarked').mean())

for dataset in data:
   dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

print(train.head())

# build model
X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']
X_test = test
X_train.shape, Y_train.shape, X_test.shape
accuracies = pd.DataFrame()

# Support vector machines
svc = SVC(kernel='poly', degree=8)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
accuracies['Support Vector Machines'] = [acc_svc]

print(accuracies['Support Vector Machines'])

print(test_df.head())


#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_pred})

#Visualize the first 5 rows
print(submission.head())

submission.to_csv('submission.csv', index=False)


