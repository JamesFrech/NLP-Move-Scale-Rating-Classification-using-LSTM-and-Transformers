import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/F1Scores.csv')
print(data)
data['ModelName'] = data['Model'].apply(lambda x: x.split('_')[-1])

tasks = data['TaskName'].unique()
print(tasks)

# single task, RFC, and ROBERTA
one_task_results = data.loc[data['NTasks']==1]
#one_task_results.set_index('Task_for_F1',inplace=True)
print(one_task_results)

#one_task_results.pivot_table(index='Task_for_F1', columns='Model', aggfunc='F1').loc[reversed(tasks)].plot.bar()
ax = plt.axes()
one_task_results.pivot_table(values='F1',
                             index='TaskName',
                             columns='ModelName').loc[reversed(tasks)].plot.bar(ax=ax,
                                                                                ylabel='F1 Score',
                                                                                xlabel='Task',
                                                                                title='Model Comparison for each Task',
                                                                                rot=45)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
plt.savefig('Images/SingleTaskF1.png',bbox_inches='tight',dpi=300)
plt.close()


# single task, 5 task, RFC, and ROBERTA
one_all_task_results = data.loc[data['NTasks'].isin([1,5])]
one_all_task_results.loc[one_all_task_results['NTasks']==5,'ModelName']='LSTM_5Task'
#one_task_results.set_index('Task_for_F1',inplace=True)
print(one_all_task_results)
#one_task_results.pivot_table(index='Task_for_F1', columns='Model', aggfunc='F1').loc[reversed(tasks)].plot.bar()
ax = plt.axes()
one_all_task_results.pivot_table(values='F1',
                             index='TaskName',
                             columns='ModelName').loc[reversed(tasks)].plot.bar(ax=ax,
                                                                                ylabel='F1 Score',
                                                                                xlabel='Task',
                                                                                title='Model Comparison for each Task',
                                                                                rot=45)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
plt.savefig('Images/SingleAllTaskF1.png',bbox_inches='tight',dpi=300)
plt.close()




SR = data.loc[data['Task_for_F1']=='SR'].sort_values('F1',ascending=True)
SR.loc[SR['Model']=='SR_P_C3_C4_A_LSTM','Model'] = '5Task_LSTM'
SR.plot.bar(x='Model',y='F1',rot=45,legend=False,title='Model Comparisons for Scale Ratings')
plt.savefig("Images/ScaleRatingF1.png",bbox_inches='tight',dpi=300)
plt.close()
print(SR)

SR = data.loc[(data['Task_for_F1']=='SR')&(data['ModelName']=='LSTM')].sort_values('F1',ascending=True)
SR.loc[SR['Model']=='SR_P_C3_C4_A_LSTM','Model'] = '5Task_LSTM'
SR.plot.bar(x='Model',y='F1',rot=45,legend=False,title='Model Comparisons for Scale Ratings',ylabel='F1 Score')
plt.savefig("Images/ScaleRatingF1_LSTMonly.png",bbox_inches='tight',dpi=300)
plt.close()
print(SR)

# Models for 2 tasks, one task, RFC, and ROBERTA
TwoTask = data.loc[data['Task_for_F1'].isin(['A','C3','C4','P'])&(data['NTasks']<=2)]
TwoTask.loc[TwoTask['NTasks']==2,'ModelName'] = 'LSTM_2Task'
tasks = ['Authorship','Polarity','Class3','Class4']
print(tasks)
ax = plt.axes()
TwoTask.pivot_table(values='F1',
                    index='TaskName',
                    columns='ModelName').loc[reversed(tasks)].plot.bar(ax=ax,
                                                                       ylabel='F1 Score',
                                                                       xlabel='Task',
                                                                       title='Model Comparison for each Task',
                                                                       rot=45)
#plt.legend(bbox_to_anchor=(1,-0.3),ncols=4)
plt.savefig("Images/TwoTaskLSTMComparison.png",bbox_inches='tight',dpi=300)
plt.close()
print(TwoTask)

# Any task number for everything except SR
TwoTask = data.loc[data['Task_for_F1'].isin(['A','C3','C4','P'])]
TwoTask.loc[TwoTask['NTasks']==2,'ModelName'] = 'LSTM_2Task'
TwoTask.loc[TwoTask['NTasks']==5,'ModelName'] = 'LSTM_5Task'
tasks = ['Authorship','Polarity','Class3','Class4']
print(tasks)
ax = plt.axes()
TwoTask.pivot_table(values='F1',
                    index='TaskName',
                    columns='ModelName').loc[reversed(tasks)].plot.bar(ax=ax,
                                                                       ylabel='F1 Score',
                                                                       xlabel='Task',
                                                                       title='Model Comparison for each Task',
                                                                       rot=45)
#plt.legend(bbox_to_anchor=(1,-0.3),ncols=4)
plt.savefig("Images/FiveTaskLSTMComparison.png",bbox_inches='tight',dpi=300)
plt.close()
print(TwoTask)



# LSTMOnly
LSTM = data.loc[data['Task_for_F1'].isin(['A','C3','C4','P'])&(data['ModelName']=='LSTM')]
LSTM.loc[LSTM['NTasks']==2,'ModelName'] = 'LSTM_2Task'
LSTM.loc[LSTM['NTasks']==5,'ModelName'] = 'LSTM_5Task'
tasks = ['Authorship','Polarity','Class3','Class4']
print(tasks)
ax = plt.axes()
LSTM.pivot_table(values='F1',
                 index='TaskName',
                 columns='ModelName').loc[reversed(tasks)].plot.bar(ax=ax,
                                                                    ylabel='F1 Score',
                                                                    xlabel='Task',
                                                                    title='Model Comparison for each Task',
                                                                    rot=45)
#plt.legend(bbox_to_anchor=(1,-0.3),ncols=4)
plt.savefig("Images/LSTMTasksComparison.png",bbox_inches='tight',dpi=300)
plt.close()
print(TwoTask)

exit()

for task in tasks:
    print(task)
    dat = data.loc[data['Task_for_F1'] == task]

    lstm_results = dat[dat['Model'].str.contains('LSTM')]
    one_task_results = dat.loc[data['NTasks']==1]
    one_task_results['Name'] = one_task_results['Model'].apply(lambda x: x.split('_')[-1])
    print(lstm_results)
    print(one_task_results)
    one_task_results.plot.bar(x='Name',y='F1',legend=False,rot=45,ylabel='F1 Score')
    plt.show()
