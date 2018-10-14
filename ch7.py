from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



df_wine = pd.read_csv('https://archive.ics.uci.edu/'

                      'ml/machine-learning-databases/wine/wine.data',

                      header=None)



# if the Wine dataset is temporarily unavailable from the

# UCI machine learning repository, un-comment the following line

# of code to load the dataset from a local path:



# df_wine = pd.read_csv('wine.data', header=None)





df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',

                   'Alcalinity of ash', 'Magnesium', 'Total phenols',

                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',

                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',

                   'Proline']



print('Class labels', np.unique(df_wine['Class label']))

df_wine.head()



X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values



X_train, X_test, y_train, y_test =    train_test_split(X, y, 

                     test_size=0.1, 

                     #random_state=0, 

                     stratify=y)


# # Assessing feature importance with Random Forests



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y)

CVS=[]
scores=cross_val_score(RandomForestClassifier(n_estimators=500),X_train,y_train,cv=10,scoring='accuracy')
print(scores)
CVS.append(scores)
pd.set_option('precision',3)
result=pd.DataFrame(CVS,columns=list(range(1,11)),)
result['mean']=result.mean(1)
result['std']=result.std(1)
## run the DecisionTree
dt=RandomForestClassifier(n_estimators=500)
dt.fit(X_train,y_train)
result['Out-of-sample-accuracy']=dt.score(X_test,y_test)
result





def display_plot(cv_scores, cv_scores_std,str):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(n_estimators_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(n_estimators_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('n_estimators')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([n_estimators_space[0], n_estimators_space[-1]])
    ax.set_xscale('log')
    plt.savefig(str+'10_10.png', dpi=300)
    plt.show()
    
######################################################################
n_estimators_space =[ 1 ,10,50,  100, 500,1000]
n_estimators_space=np.arange(100)
n_estimators_space=np.linspace(1,1000, dtype = int, endpoint=False, num=20)
n_estimators_space=np.logspace(0, 3, 10,dtype=int)
rf_scores = []
rf_scores_std = []

rfclass=RandomForestClassifier(n_estimators=500)
# Compute scores over range of alphas
for alpha in n_estimators_space:

    # Specify the alpha value to use: ridge.alpha
    rfclass.n_estimators=alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    rf_cv_scores = cross_val_score(rfclass, X_train, y_train, cv=10,scoring='accuracy')
    
    # Append the mean of ridge_cv_scores to ridge_scores
    rf_scores.append(np.mean(rf_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    rf_scores_std.append(np.std(rf_cv_scores))

# Display the plot
display_plot(rf_scores, rf_scores_std,'rf')  


result2=pd.DataFrame(CVS,columns=n_estimators_space,index=["mean","std"])
CVS
CVS=[]
CVS.append(rf_scores)
CVS.append(rf_scores_std)

result2


# # Assessing feature importance with Random Forests





################################################################################



feat_labels = df_wine.columns[1:]



forest = RandomForestClassifier()

param_rf={
        'n_estimators':n_estimators_space
}
gs=GridSearchCV(estimator=forest,
                param_grid=param_rf,
                scoring='accuracy',
                cv=10,
                n_jobs=-1)

gsm=gs.fit(X_train, y_train)

forest=gsm.best_estimator_

print(gsm.best_params_)

importances = forest.feature_importances_



indices = np.argsort(importances)[::-1]


for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30, 

                            feat_labels[indices[f]], 

                            importances[indices[f]]))



plt.title('Feature Importance')

plt.bar(range(X_train.shape[1]), 

        importances[indices],

        align='center')



plt.xticks(range(X_train.shape[1]), 

           feat_labels[indices], rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()

#plt.savefig('images/04_09.png', dpi=300)

plt.show()











sfm = SelectFromModel(forest, threshold=0.1, prefit=True)

X_selected = sfm.transform(X_train)

print('Number of features that meet this threshold criterion:', 

      X_selected.shape[1])





# Now, let's print the 3 features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):







for f in range(X_selected.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30, 

                            feat_labels[indices[f]], 

                            importances[indices[f]]))







# # Summary



# ...



# ---

# 

# Readers may ignore the next cell.

#####################################################################
#forest = RandomForestRegressor(n_estimators=1000, 
#                               criterion='mse', 
#                               random_state=1, 
#                               n_jobs=-1)
#forest.fit(X_train, y_train)
#y_train_pred = forest.predict(X_train)
#y_test_pred = forest.predict(X_test)
#
#print('MSE train: %.3f, test: %.3f' % (
#        mean_squared_error(y_train, y_train_pred),
#        mean_squared_error(y_test, y_test_pred)))
#print('R^2 train: %.3f, test: %.3f' % (
#        r2_score(y_train, y_train_pred),
#        r2_score(y_test, y_test_pred)))
#
#
#
#
#plt.scatter(y_train_pred,  
#            y_train_pred - y_train, 
#            c='steelblue',
#            edgecolor='white',
#            marker='o', 
#            s=35,
#            alpha=0.9,
#            label='training data')
#plt.scatter(y_test_pred,  
#            y_test_pred - y_test, 
#            c='limegreen',
#            edgecolor='white',
#            marker='s', 
#            s=35,
#            alpha=0.9,
#            label='test data')
#
#plt.xlabel('Predicted values')
#plt.ylabel('Residuals')
#plt.legend(loc='upper left')
#plt.hlines(y=0, xmin=0, xmax=10, lw=2, color='black')
#plt.xlim([0, 5])
#plt.tight_layout()
#
## plt.savefig('images/10_14.png', dpi=300)
#plt.show()
#
#
#
#
