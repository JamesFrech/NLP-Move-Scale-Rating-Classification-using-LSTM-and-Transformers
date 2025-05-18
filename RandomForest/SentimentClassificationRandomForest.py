import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

train = pd.read_csv('Data/CornellMovieReview_scale_data_train_with_bigramunigrams.csv')
val = pd.read_csv('Data/CornellMovieReview_scale_data_val_with_bigramunigrams.csv')
test = pd.read_csv('Data/CornellMovieReview_scale_data_test_with_bigramunigrams.csv')

print(train)
input_cols = [i for i in train.columns if i not in ['Review','Rating','ID','Author']]

train['Rating'] = train['Rating']*10
val['Rating'] = val['Rating']*10
test['Rating'] = test['Rating']*10

# Use optimal parameters
rf = RandomForestClassifier(n_estimators = 600, max_features = 75, min_samples_leaf = 3)
rf.fit(train[input_cols],train['Rating'])
train['pred'] = rf.predict(train[input_cols])


print('Train F1:', f1_score(train['Rating'],train['pred'],average='weighted'))

val['pred'] = rf.predict(val[input_cols])
print('Validation F1:', f1_score(val['Rating'],val['pred'],average='weighted'))

print('Feature Importances')
feature_importance = pd.DataFrame(rf.feature_importances_, index = input_cols, columns = ['FI'])
feature_importance.sort_values('FI',ascending=False,inplace=True)
print(feature_importance.loc[feature_importance['FI']>0.01])

cm = confusion_matrix(val['Rating'], val['pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest Validation Results")
plt.savefig('Images/RFC_ConfusionMatrix.png',bbox_inches='tight')
plt.close()



test['pred'] = rf.predict(test[input_cols])
print('Test F1:', f1_score(test['Rating'],test['pred'],average='weighted'))

cm = confusion_matrix(test['Rating'], test['pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
#plt.title("Random Forest Test Results")
plt.savefig('Images/RFC_ConfusionMatrix_Test.png',bbox_inches='tight')
plt.close()
