import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

train = pd.read_csv('Data/CornellMovieReview_scale_data_train_with_bigramunigrams.csv')
val = pd.read_csv('Data/CornellMovieReview_scale_data_val_with_bigramunigrams.csv')
test = pd.read_csv('Data/CornellMovieReview_scale_data_test_with_bigramunigrams.csv')

target='Class4'

# Multiply scale ratings by 10 for integers
train['Rating'] = train['Rating']*10
val['Rating'] = val['Rating']*10
test['Rating'] = test['Rating']*10

# Create variables not in current version of dataset
train['Polarity'] = [0]*train.shape[0]
val['Polarity'] = [0]*val.shape[0]
test['Polarity'] = [0]*test.shape[0]
train['Class3'] = [0]*train.shape[0]
val['Class3'] = [0]*val.shape[0]
test['Class3'] = [0]*test.shape[0]
train['Class4'] = [0]*train.shape[0]
val['Class4'] = [0]*val.shape[0]
test['Class4'] = [0]*test.shape[0]

# Make polarity variable
train.loc[train['Rating'] > 5,'Polarity'] = 1
val.loc[val['Rating'] > 5,'Polarity'] = 1
test.loc[test['Rating'] > 5,'Polarity'] = 1

# Make class3 variable
train.loc[train['Rating'] <= 4,'Class3'] = 0
train.loc[(train['Rating'] > 4) & (train['Rating'] < 7),'Class3'] = 1
train.loc[train['Rating'] >= 7,'Class3'] = 2
val.loc[val['Rating'] <= 4,'Class3'] = 0
val.loc[(val['Rating'] > 4) & (val['Rating'] < 7),'Class3'] = 1
val.loc[val['Rating'] >= 7,'Class3'] = 2
test.loc[test['Rating'] <= 4,'Class3'] = 0
test.loc[(test['Rating'] > 4) & (test['Rating'] < 7),'Class3'] = 1
test.loc[test['Rating'] >= 7,'Class3'] = 2

# Make class4 variable
train.loc[train['Rating'] <= 3,'Class4'] = 0
train.loc[(train['Rating'] >= 4) & (train['Rating'] <= 5),'Class4'] = 1
train.loc[(train['Rating'] >= 6) & (train['Rating'] <= 7),'Class4'] = 2
train.loc[train['Rating'] >= 8,'Class4'] = 3
val.loc[val['Rating'] <= 3,'Class4'] = 0
val.loc[(val['Rating'] >= 4) & (val['Rating'] <= 5),'Class4'] = 1
val.loc[(val['Rating'] >= 6) & (val['Rating'] <= 7),'Class4'] = 2
val.loc[val['Rating'] >= 8,'Class4'] = 3
test.loc[test['Rating'] <= 3,'Class4'] = 0
test.loc[(test['Rating'] >= 4) & (test['Rating'] <= 5),'Class4'] = 1
test.loc[(test['Rating'] >= 6) & (test['Rating'] <= 7),'Class4'] = 2
test.loc[test['Rating'] >= 8,'Class4'] = 3

print(train)
input_cols = [i for i in train.columns if i not in ['Review','Rating','ID','Author','Class3','Polarity','Class4']]

#train['Rating'] = train['Rating']*10
#val['Rating'] = val['Rating']*10
#test['Rating'] = test['Rating']*10

# Use optimal parameters
rf = RandomForestClassifier(n_estimators = 600, max_features = 75, min_samples_leaf = 3)
rf.fit(train[input_cols],train[target])
train['pred'] = rf.predict(train[input_cols])


print('Train F1:', f1_score(train[target],train['pred'],average='weighted'))

val['pred'] = rf.predict(val[input_cols])
print('Validation F1:', f1_score(val[target],val['pred'],average='weighted'))

print('Feature Importances')
feature_importance = pd.DataFrame(rf.feature_importances_, index = input_cols, columns = ['FI'])
feature_importance.sort_values('FI',ascending=False,inplace=True)
print(feature_importance.loc[feature_importance['FI']>0.01])

test['pred'] = rf.predict(test[input_cols])
print('Test F1:', f1_score(test[target],test['pred'],average='weighted'))

cm = confusion_matrix(test[target], test['pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Random Forest Test Results: {target}")
plt.savefig(f'Images/RFC_ConfusionMatrix_Test_{target}.png',bbox_inches='tight')
plt.close()
