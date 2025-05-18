import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

train = pd.read_csv('Data/CornellMovieReview_scale_data_train_with_bigramunigrams.csv')
val = pd.read_csv('Data/CornellMovieReview_scale_data_val_with_bigramunigrams.csv')

print(train)
input_cols = [i for i in train.columns if i not in ['Review','Rating','ID','Author']]

train['Rating'] = train['Rating']*10
val['Rating'] = val['Rating']*10

rf = RandomForestClassifier()
rf.fit(train[input_cols],train['Rating'])
train['pred'] = rf.predict(train[input_cols])


print('Train F1:', f1_score(train['Rating'],train['pred'],average='weighted'))

val['pred'] = rf.predict(val[input_cols])
print('Validation F1:', f1_score(val['Rating'],val['pred'],average='weighted'))

print('Feature Importances')
feature_importance = pd.DataFrame(rf.feature_importances_, index = input_cols, columns = ['FI'])
feature_importance.sort_values('FI',ascending=False,inplace=True)
print(feature_importance.loc[feature_importance['FI']>0.01])

# Use 5 fold cross validation
all_train = pd.concat([train,val])
n = all_train.shape[0]
all_train['fold'] = [0]*all_train.shape[0]
all_train['fold'].iloc[int(n/5):int(2*n/5)] = 1
all_train['fold'].iloc[int(2*n/5):int(3*n/5)] = 2
all_train['fold'].iloc[int(3*n/5):int(4*n/5)] = 3
all_train['fold'].iloc[int(4*n/5):] = 4

#i=0
for trees in [600,700,800,900]:
    for n_features in [100,50,75]:
        for min_samples in [2,3,4]:
            print(trees,n_features,min_samples)
            f1 = []
            for fold in [0,1,2,3,4]:
                # Get train and validation splits
                train = all_train.loc[all_train['fold'] != fold]
                val = all_train.loc[all_train['fold'] == fold]

                # Initialize model with sets of hyperparameters
                rf = RandomForestClassifier(n_estimators = trees,
                                            max_features = n_features,
                                            min_samples_leaf = min_samples)

                rf.fit(train[input_cols],train['Rating'])
                pred = rf.predict(val[input_cols])

                f1_fold = f1_score(val['Rating'], pred, average = 'weighted')
                f1.append(f1_fold)

            f1 = np.mean(f1)
            results = pd.DataFrame(np.array([[trees,n_features,min_samples,f1]]),
                                   columns = ['nTrees','nFeatures','minSamples','F1'])
            #if i==0:
                #results.to_csv('Data/results/RFCrossVal.csv',index=False)
                #i+=1
            #else:
            results.to_csv('Data/results/RFCrossVal.csv',index=False,header=False,mode='a')
