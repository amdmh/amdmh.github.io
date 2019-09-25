---
layout: post
title:  "Spotify Music Analysis - Part 4 "
comments : true
tags: API spotify spotipy python machine-learning
---

```python
#Import packages
import numpy as np
import numpy.core.multiarray
import pandas as pd
import sklearn
import shap
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```


```python
X = pd.read_pickle('data/X.pkl')
Y = pd.read_pickle('data/Y.pkl')
```


```python
X.shape
```




    (1000, 20)




```python
#Keep a copy of the full dataframe that we'll use later
X_complete = X.copy()
```


```python
X.drop(columns=['track_name', 'artist_name'], inplace=True)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size = 0.2,
                                                    train_size= 0.8,
                                                    random_state = 0)
```

## Baseline model ( without tuning hyperparameters)


```python
model = RandomForestClassifier(n_estimators=200,random_state=0)
```


```python
model.fit(X_train, y_train.values.ravel())
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_jobs=None, oob_score=False, random_state=0, verbose=0,
                           warm_start=False)




```python
predictions = model.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predictions) * 100
rounded_score = round(score, 1)
```


```python
print("Random Forest Classifier Accuracy : {}%".format(rounded_score))
```

    Random Forest Classifier Accuracy : 82.0%
    

## Using a randomized search to improve the model accuracy


```python
from scipy.stats import randint as sp_randint

#Set parameters
param_grid = {
    'n_estimators': sp_randint(100, 1200),
    'bootstrap': [False, True],
    'max_depth':sp_randint(3,30),
    'min_samples_leaf': sp_randint(2,10),
    'min_samples_split': sp_randint(2,100),
    "bootstrap": [True, False],
    'max_features': ['auto', 'log2', 'sqrt'],
    'criterion': ['gini', 'entropy']
    
}
```


```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
```


```python
randomCV = RandomizedSearchCV(RandomForestClassifier(), random_state= 0, param_distributions=param_grid,
                              verbose=3, n_jobs=7, refit=False, n_iter=100)
randomCV.fit(X_train, y_train)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    

    [Parallel(n_jobs=7)]: Using backend LokyBackend with 7 concurrent workers.
    [Parallel(n_jobs=7)]: Done  18 tasks      | elapsed:    2.4s
    [Parallel(n_jobs=7)]: Done 114 tasks      | elapsed:   23.1s
    [Parallel(n_jobs=7)]: Done 274 tasks      | elapsed:   55.2s
    [Parallel(n_jobs=7)]: Done 300 out of 300 | elapsed:   58.6s finished
    




    RandomizedSearchCV(cv='warn', error_score='raise-deprecating',
                       estimator=RandomForestClassifier(bootstrap=True,
                                                        class_weight=None,
                                                        criterion='gini',
                                                        max_depth=None,
                                                        max_features='auto',
                                                        max_leaf_nodes=None,
                                                        min_impurity_decrease=0.0,
                                                        min_impurity_split=None,
                                                        min_samples_leaf=1,
                                                        min_samples_split=2,
                                                        min_weight_fraction_leaf=0.0,
                                                        n_estimators='warn',
                                                        n_jobs=None,
                                                        o...
                       param_distributions={'bootstrap': [False, True],
                                            'criterion': ['gini', 'entropy'],
                                            'max_depth': [3, 5, 8, 15, 25, 30],
                                            'max_features': ['auto', 'log2',
                                                             'sqrt'],
                                            'min_samples_leaf': [2, 5, 10],
                                            'min_samples_split': [2, 5, 10, 15,
                                                                  100],
                                            'n_estimators': [100, 200, 300, 500,
                                                             800, 1200]},
                       pre_dispatch='2*n_jobs', random_state=0, refit=False,
                       return_train_score=False, scoring=None, verbose=3)




```python
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
```


```python
report(randomCV.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.844 (std: 0.018)
    Parameters: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 15, 'criterion': 'entropy', 'bootstrap': False}
    
    Model with rank: 2
    Mean validation score: 0.836 (std: 0.014)
    Parameters: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 25, 'criterion': 'entropy', 'bootstrap': False}
    
    Model with rank: 2
    Mean validation score: 0.836 (std: 0.015)
    Parameters: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 30, 'criterion': 'entropy', 'bootstrap': False}
    
    

## Is normalisation improve the model accuary ?


```python
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)
```


```python
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, 
                                                                                y_train,
                                                                                test_size = 0.2,
                                                                                train_size= 0.8,
                                                                                random_state=0)
```


```python
rf = RandomForestClassifier(n_estimators=200,random_state=0)
```


```python
rf.fit(X_train_sub, y_train_sub.values.ravel())
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_jobs=None, oob_score=False, random_state=0, verbose=0,
                           warm_start=False)




```python
preds = rf.predict(X_validation_sub)
```


```python
from sklearn.metrics import accuracy_score
score = accuracy_score(y_validation_sub, predictions) * 100
rounded_score = round(score, 1)
```


```python
print("Random Forest Classifier acuracy score on testing data: {}%".format(rounded_score))
```

    Random Forest Classifier acuracy score on testing data: 84.4%
    

Considering the small sample used to drive the model, it has rather good performance (also knowing that the optimization of hyper-parameters has not been pushed to the limit to limit computation time and excessive memory usage) ranging between 82 and 84% accuracy depending on the case

## Interpreting the model with SHAP

Considering everything that has been done so far, I thought it would be interesting to explain which features mattered to my model and make comparisons between them. That's why I used SHAP which provides a unified approach for intepreting output of machine learning methods. 


```python
shap.initjs()
```

```python
shap_values = shap.TreeExplainer(model).shap_values(X)
```


```python
X_importance = X
```


```python
# Explain model predictions using shap library:
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
```


```python
# Plot summary_plot
shap.summary_plot(shap_values, X_importance)
```


![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/spotify/output_35_0.png)


The shap_values object above is a list with two arrays. The first array is the SHAP values for a negative outcome (Class 0 means song that I don't like), and the second array is the list of SHAP values for the positive outcome (Class 1 means song that I actually like). Popularity, energy and age are three most importances features : this is not surprising because those features emerged as the most significant to represent my musical tastes in the previous step. 


```python
row_to_show = 161
row_for_prediction = X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
row_for_prediction_array = row_for_prediction.values.reshape(1, -1)
model.predict_proba(row_for_prediction_array)
```




    array([[0.025, 0.975]])



Based on the trained model, there's a 97% chance that I like the song - which is If I ruled the world by Nas & Lauryn Hill -


```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(row_for_prediction)
```


```python
# use Tree SHAP to explain predictions
shap.force_plot(explainer.expected_value[1], shap_values[1], row_for_prediction)
```





![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/spotify/value_row.PNG)

**N.B.: Due to the lack of dynamic display, it is impossible to present a graph allowing a comparison of the most important variables, line by line.**

To conclude the work that has been done, it is important to remember that despite more than correct accuracy, the model is biased by the fact that it was very difficult to select a representative and as diversified as possible sample for the songs that I did not like, a problem that obviously does not exist for the songs that I like. 

Furthermore, the exploratory analysis could be further developed by creating a script dedicated to the search for patterns characterizing the songs I like. In the same way, one of the ways to improve would be to compare several machine learning models (AdaBoostClassifier or GradientBoostingClassifier for example) to retain only the best. For now, I've decided to take a break from the subject by exploring a new angle: the prediction of a song's success, i.e. whether or not it will reach the Billboard Hot 100 Hits. 
