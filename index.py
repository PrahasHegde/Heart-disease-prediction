#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix



df = pd.read_csv('dataset_heart.csv')
pd.set_option('display.max_columns', 100)
print(df.head())
print(df.shape)
print(df.info())

y = df['heart disease']
X = df.drop(columns='heart disease')


#using cross validation score
"""In the cross_val_score() method I need to pass 5 main things.Model,Features,Target/Label,Number of folds (For example, if you have 100 rows in your dataset in the first round the
first 80 rows will be used for training and the remaining 20 will be used for testing. 
In the second round, the first 60(1–60) and the last 20(81–100) will be used for training and the rest will 
be used for training and so on. This happens 5 times(as I mentioned cv=5) and returns us 5 different scores we can get 
the mean of it using the mean() method).
Scoring (In the scoring I need to choose an evaluation metric which can be accuracy, precision score, recall score, f1 score, and others.In this case, I used accuracy.
"""

# dtc_scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5, scoring='accuracy')
# print(dtc_scores) 
# print(dtc_scores.mean())

"""The main goal of using validation scores is to evaluate the model’s performance on unseen data.
 This helps to check for overfitting and underfitting
"""


models = [DecisionTreeClassifier(), RandomForestClassifier()]
for model in models:
  scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
  print(model, scores.mean()) 


#grid search
"""Grid search is used to find the best hyperparameters for a machine learning model. 
Hyperparameters are parameters that are set before training the model, and they control the model’s learning process.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)

#model
#It is telling me to set max_depth=10 for better accuracy let’s try it.
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_prediction = rfc.predict(X_test)
rfc_score = accuracy_score(y_test, rfc_prediction)
print(rfc_score) #90% accuracy


#confusion matrix
confmat = confusion_matrix(y_test,rfc_prediction)
print(confmat)
# Plotting the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
