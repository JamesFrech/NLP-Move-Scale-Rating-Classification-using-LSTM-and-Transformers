import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('Data/results/RFCrossVal.csv')

nFeatures = [100,50,75]
minSamples = [2,3,4]

for n in nFeatures:
    data = results.loc[results['nFeatures']==n]
    data = data.groupby(['nTrees','nFeatures'])['F1'].mean().reset_index()
    plt.plot(data['nTrees'],data['F1'],label=f'{n} features')
plt.xlabel('Number of Trees')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 score for Number of Trees and Number of Features at Split')
plt.savefig('Images/nTrees_nfeatures_F1.png',bbox_inches='tight')
plt.close()

for n in minSamples:
    data = results.loc[results['minSamples']==n]
    data = data.groupby(['nTrees','minSamples'])['F1'].mean().reset_index()
    plt.plot(data['nTrees'],data['F1'],label=f'Min Samples Leaf: {n}')
plt.xlabel('Number of Trees')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 score for Number of Trees and Minimum Samples per Leaf')
plt.savefig('Images/nTrees_minSamples_F1.png',bbox_inches='tight')
plt.close()
