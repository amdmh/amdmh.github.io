---
layout: post
title:  "Udacity Data Science Nanodegree : Finding Donors for CharityML"
tags: udacity python machine-learning supervised-learning
---

## Getting Started

In this project, we will employ several supervised algorithms of our choice to accurately model individuals' income using data collected from the 1994 U.S. Census. We will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Our goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features.

The dataset for this project originates from the [UCI Machine Learning Repository] (https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

----
## 1 - Exploring the Data


```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))
```
<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/finding_donors/HEAD_1.PNG"  width="673" height="59">


### 1.1 - Implementation: Data Exploration

```python
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
```

    45222
    11208
    Total number of records: 45222
    Individuals making more than $50,000: 11208
    Individuals making at most $50,000: 34014
    Percentage of individuals making more than $50,000: 24.78%
    

** Featureset Exploration **

* **age**: continuous. 
* **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
* **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
* **education-num**: continuous. 
* **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
* **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
* **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
* **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
* **sex**: Female, Male. 
* **capital-gain**: continuous. 
* **capital-loss**: continuous. 
* **hours-per-week**: continuous. 
* **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

----
## 2 - Preparing the Data

### 2.1 - Transforming Skewed Continuous Features

```python
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)
```


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/finding_donors/output_9_0.png"  width="706" height="337">



For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.



```python
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)
```


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/finding_donors/output_11_0.png"  width="706" height="337">


### 2.2 - Normalizing Numerical Features


```python
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))
```
<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/finding_donors/HEAD_2.PNG"  width="821" height="191">

### 2.3 - Implementation: Data Preprocessing

From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.

|   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
| :-: | :-: |                            | :-: | :-: | :-: |
| 0 |  B  |  | 0 | 1 | 0 |
| 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
| 2 |  A  |  | 1 | 0 | 0 |


```python
# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 0 if x=='<=50K' else 1)

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
#print(encoded)
```

    103 total features after one-hot encoding.
    

### 2.4 - Shuffle and Split Data


```python
# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
```

    Training set has 36177 samples.
    Testing set has 9045 samples.
    

    /opt/conda/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

----
## 3 - Evaluating Model Performance
In this section, we will investigate four different algorithms, and determine which is best at modeling the data. 

### 3.1 - Metrics and the Naive Predictor
*CharityML*, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that *does not* make more than \$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:

$$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$

In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).

Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, *CharityML* would identify no one as donors. 


#### Note: Recap of accuracy, precision, recall

** Accuracy ** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).

** Precision ** tells us what proportion of messages we classified as spam, actually were spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of

`[True Positives/(True Positives + False Positives)]`

** Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of

`[True Positives/(True Positives + False Negatives)]`

For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average(harmonic mean) of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score(we take the harmonic mean as we are dealing with ratios).

### Question 1 - Naive Predictor Performace


```python
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case

# TODO: Calculate accuracy, precision and recall
accuracy = TP / (TP + FP)
recall = TP / (TP + FN)
precision = TP / (TP + FP)

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = (1+beta**2)*(accuracy*recall)/(beta**2*accuracy+recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
```

    Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2917]
    

### Supervised Learning Models

### Question 2 - Model Application
List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen

- Describe one real-world application in industry where the model can be applied. 
- What are the strengths of the model; when does it perform well?
- What are the weaknesses of the model; when does it perform poorly?
- What makes this model a good candidate for the problem, given what you know about the data?

### 1 - Random Forest

**Real world applications**

- Random Forest can be used for a plethora of data circumstances including : 
    - Image classification
    - Detecting fraudulent cases in banking systems
    - Recommendation engines
    
**Strenghts**

- Works well on very large datasets
- Highly accurate and robust method (multiple decision trees)
- Fast to train
- Provides an estimation of the most important features in the classification
- No pre-processing required
- Robust to outliers
- Can handle missing values

**Weaknesses**

- Can become slow ( = high prediction time) on large data sets (multiple decision trees)
- Overfitting in case of noisy data
- Less interpretable


**Why choosing this model**

Given the size of the dataset (14 variables, 45000 lines), the robustness and very high average accuracy of the model for most cases, random forest is usually a safe choice. Beyond its advantages, this model was mainly used to benchmark its efficiency in the face of logistic regression.


### 2 - Logistic Regression

**Real word applications**

- This model is widely used in many fields like : 

    - **Marketing** : a marketing consultant wants to predict if the subsidiary of his company will make profit, loss or just break even depending on the characteristic of the subsidiary operations

    - **Human Resources** : the HR manager of a company wants to predict the absenteeism pattern of his employees based on their individual characteristic.

    - **Finance** : a bank wants to predict if his customers would default based on the previous transactions and history.

**Strenghts** 

- Easy to use and implement
- Fast to train
- Has a low variance 
- Returns probability outcomes (wich can be useful for ranking purposes)

**Weaknesses**

- Suffers from high biais
- Bad when too many features or too many classifications
- Requires transformations for non-linear features
- Tends to underperform when there are multiple or non-linear decision boundaries. 
- Not flexible enough to naturally capture more complex relationships.

**Why choosing this model**

Even if it wasn't my first choice, I think logistic regression, which is considered as one of the most popular techniques for binary classification problems, fits to our case (we are predicting a categorical response which is whether or not an individual makes more than $50,000 dollars per year and so, is more willing to donate to CharityML). 


### 3 - Gradient Boosting Classifier

**Real word applications**

- Gradient boosting can be used in the field of learning to rank : the commercial web search engines Yahoo and Yandex use variants of gradient boosting in their machine-learned ranking engines

**Strenghts**

- Predictive power
- Flexible (can optimize on different loss functions and provides several hyperparameter tuning options)
- Robustness to outliers in output space (via robust loss functions)
- No data pre-processing required 
- Natural handling of data of mixed type (= heterogeneous features)
- Handles missing data (imputation not required)

**Weaknesses**

- More sensitive to overfitting if the data is noisy
- Slow in traning (trees are built sequentially)
- Harder to tune compared to Random Forest 
- Computationally expensive
- Less interpretable
- Scalability, due to the sequential nature of boosting it can hardly be parallelized

**Why choosing this model**

Ensemble methods and Gradient boosted machines (GBMs) in particular are an extremely popular machine learning algorithm that have proven successful across many domains and is one of the leading methods for winning Kaggle competitions. Note that we have an heterogeneous dataset so GBM is completely suitable. 

### 3.2 - Implementation - Creating a Training and Predicting Pipeline


```python
# Import two metrics from sklearn - fbeta_score and accuracy_score

from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end-start
        
    #       Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300],predictions_train,0.5)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test,predictions_test,0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results
```

### 3.3 - Implementation: Initial Model Evaluation


```python
# Import the three supervised learning models from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Initialize the three models
clf_A = RandomForestClassifier(random_state = 50)
clf_B = LogisticRegression(random_state = 50)
clf_C = GradientBoostingClassifier(random_state = 50)

# Calculate the number of samples for 1%, 10%, and 100% of the training data

samples_100 = len(y_train)
samples_10 = int(len(y_train)/10)
samples_1 = int(len(y_train)/100)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)
```

    RandomForestClassifier trained on 361 samples.
    RandomForestClassifier trained on 3617 samples.
    RandomForestClassifier trained on 36177 samples.
    LogisticRegression trained on 361 samples.
    LogisticRegression trained on 3617 samples.
    LogisticRegression trained on 36177 samples.
    GradientBoostingClassifier trained on 361 samples.
    GradientBoostingClassifier trained on 3617 samples.
    GradientBoostingClassifier trained on 36177 samples.
    


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/finding_donors/output_28_1.png"  width="706" height="502">




```python
#Results summary
for i in results.items():
    print(i[0])
    display(pd.DataFrame(i[1]).rename(columns={0:'1%', 1:'10%', 2:'100%'}))
```

    RandomForestClassifier
    


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.808402</td>
      <td>0.829298</td>
      <td>0.837590</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.976667</td>
      <td>0.983333</td>
      <td>0.973333</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.599353</td>
      <td>0.654794</td>
      <td>0.670981</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.978916</td>
      <td>0.977011</td>
      <td>0.959302</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.024743</td>
      <td>0.030806</td>
      <td>0.047916</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.021036</td>
      <td>0.058833</td>
      <td>0.890773</td>
    </tr>
  </tbody>
</table>
</div>


    LogisticRegression
    


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.818684</td>
      <td>0.838585</td>
      <td>0.841902</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.860000</td>
      <td>0.846667</td>
      <td>0.846667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.628860</td>
      <td>0.677507</td>
      <td>0.683165</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.738636</td>
      <td>0.703125</td>
      <td>0.698529</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.004317</td>
      <td>0.004014</td>
      <td>0.003978</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.003392</td>
      <td>0.029592</td>
      <td>0.416446</td>
    </tr>
  </tbody>
</table>
</div>


    GradientBoostingClassifier
    


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.826976</td>
      <td>0.855943</td>
      <td>0.863018</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.940000</td>
      <td>0.883333</td>
      <td>0.856667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.648672</td>
      <td>0.721604</td>
      <td>0.739534</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.937500</td>
      <td>0.813492</td>
      <td>0.734127</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.033803</td>
      <td>0.027983</td>
      <td>0.034791</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.099762</td>
      <td>0.910830</td>
      <td>11.928938</td>
    </tr>
  </tbody>
</table>
</div>


----
## 4 - Improving Results

### Question 3 - Choosing the Best Model

In my opinion, the best performing and appropriate model (taking into account our case wich is identifying individudals who earn more than $50,000) is the **Gradient Boosting Classifier**. By looking at the results (plots and tables), here is what we can notice : 

- **Accuracy** : Although Random Forest has the highest accuracy on training sets, Gradient Boosting Classifier performs slightly better across all when it comes to testing

- **Time** : Gradient boosting is slower than Logistic Regression for training but does a little better in case of predicting (even if, with around 0.04 seconds, it's not the fastest) 

- **F-score** : same observation as for the accuracy where Random Forest does better on training, Gradient Boosting produce the highest result around 73% on the 100% testing dataset (compared to 68% for Logistic Regression and 67% for Random Forest) 

Finally, I would conclude by saying that, when weighing the pros and cons with regard to all attributes, I'm sticking to my first choice because Gradient Boosting gives fairly good results (in particular regarding the F-score resulting in good performance in terms of Recall and Precision) and, with regards of numbers of records, 13 is an acceptable execution time.

### Question 4 - Describing the Model in Layman's Terms

**To describe and explain Gradient Boosting Classifier in detail, we must first understand the principle of a decision tree model**

A tree model is like the Twenty Questions guessing game. The guesser might have questions like «Is it bigger than a bread box?», «Is it alive?», etc. The size or lifeness of the thing being guessed at is a feature. By winnowing down what is likely or unlikely based on these questions, you end up with a likely (but possibly wrong) answer. Part of the strategy in 20 questions is to order the questions correctly : the first few questions should be broad, so as to eliminate large number of possibilities. The last few questions should be more specific to hone in on the «best» possible answer. 

Now, what happens when a Tree ML is trained on the data set, the algorithm tries to come up with a set of «questions» that are «optimal». Unfortunately, there is no perfect solution. So, there are different strategies to try to build the Tree Model. GBC is one of the tree models.



**Explanation of GBC**

Gradient boosting is a type of machine learning boosting. It relies on the intuition that the best possible next model, when combined with previous models, **minimizes the overall prediction error**. The key idea is to **set the target outcomes for this next model in order to minimize the error**. How are the targets calculated? The target outcome for each case in the data depends on how much changing that case’s prediction impacts the overall prediction error:

- If a small change in the prediction for a case causes a large drop in error, then next target outcome of the case is a high value. Predictions from the new model that are close to its targets will reduce the error
- If a small change in the prediction for a case causes no change in error, then next target outcome of the case is zero. Changing this prediction does not decrease the error.

The name gradient boosting arises because target outcomes for each case are set based on the gradient of the error with respect to the prediction. Each new model takes a step in the direction that minimizes prediction error, in the space of possible predictions for each training case.


**Illustration for GBC**

> Imagine 20 teams (trees). A boss at the top, then subordinates, then more subordinates, and so on. Team members are explanatory variables. Assume, Trees = 20 and Depth (number of members in each team) = 5. 

So each team will have 5 members, and total members = 100. We give them a book to read, and then they will have to answer 20 questions (Number of observations in our data). Assume they have binary answers: Yes or No (in our case, less than 50k or greater). Now, we start the process. The aim of the process is to maximum correct answers by building 20 teams having 5 members each. Any member can be a part of more than 1 team, and any member can have more than 1 more than 1 role in same team. The member which have maximum roles is the most important variable of our model.

The process starts with a random guess of answers. Then it calculates error = Actual - Predicted Answer. Next step, it build a team of 5 members, which reduces the error by maximum. Again, it calculates the error. The second team (tree) has to reduce it further. But next team doesn't trust its previous partner fully, so it assume that answers are correct with x probability (learning rate). This process go on till 20 teams are build. So in the process, we have to decide, how many teams to build (trees), members in each team(depth) and learning team, so that error in the end is minimum. This can only be done by trial and error method.


### 4.1 - Implementation: Model Tuning


```python
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, r2_score, fbeta_score

# Initialize the classifier
clf = GradientBoostingClassifier(random_state=50)

# Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {
    'n_estimators':[300,400,500],
    'min_samples_split':[4],
    'max_depth':[3],
}

# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

print(best_clf)
```

    Unoptimized model
    ------
    Accuracy score on testing data: 0.8630
    F-score on testing data: 0.7395
    
    Optimized Model
    ------
    Final accuracy score on the testing data: 0.8718
    Final F-score on the testing data: 0.7545
    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=4,
                  min_weight_fraction_leaf=0.0, n_estimators=500,
                  presort='auto', random_state=50, subsample=1.0, verbose=0,
                  warm_start=False)
    


```python
#Compute area under the curve

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
fpr_gb, tpr_gb, _ = roc_curve(y_test, predictions)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))
```

    Area under ROC curve = 0.78
    

### Question 5 - Final Model Evaluation

* What is your optimized model's accuracy and F-score on the testing data? 
* Are these scores better or worse than the unoptimized model? 
* How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  

**Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

#### Results:

|     Metric     | Unoptimized Model | Optimized Model |
| :------------: | :---------------: | :-------------: | 
| Accuracy Score |        0.8630           | 0.8718                |
| F-score        |        0.7395           | 0.7545      |


- The scores of optimized models are slightly better than the unoptimized ones (maybe the tuning could have been improved)
- The results of GBC are much better than those of the Naive predictor (Accuracy increased by 0.624, F-Score by 0.4628)

----
## 5 - Feature Importance

### Question 6 - Feature Relevance Observation

When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

At first glance, I would say that the five most important features for prediction are : 

1. **capital gain** : the more an individual's capital increases, the more his/her ability to invest (or give money) increases 
2. **education** : according to several economic and sociological studies, the most highly educated people have on average a higher salary
3. **age** : in general, a person tends to gain experience over the years. As a result, older people are more likely to have salary increases (bonuses, promotions to positions with more responsibility, etc.)
4. **occupation** : the sector of activity and the hierarchical position in the company can have an influence on an individual's salary and therefore on his ability to give money
5. **race** : dozens of socio-economic studies have proven the influence of race on the economic situation and in particular on its income

### 5.1 - Implementation - Extracting Feature Importance

```python
# Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=50)

# Train the supervised model on the training set using .fit(X_train, y_train)
model = model.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)
```


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/finding_donors/output_45_0.png"  width="576" height="317">

### Question 7 - Extracting Feature Importance

* How do these five features compare to the five features you discussed in **Question 6**?
* If you were close to the same answer, how does this visualization confirm your thoughts? 
* If you were not close, why do you think these features are more relevant?

- First of all, I am quite surprised by the result and this even though my theory is partly true (at least for three of the five variables). In fact, I thought that education and occupation (which does not come out) even more would have more influence on income. 

- Another point is that I was not expecting capital loss plays such an important role but, after some thinking, it does make sense since capital loss is just the other side of the coin of capital gain. That's why losses and gains can clarify an individual's overall financial situation and therefore his or her salary. Same thing for hours-per-week even if common sense would suggest that people who work more every week are more likely to earn more money. 

### 5.2 - Feature Selection


```python
# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
```

    Final Model trained on full data
    ------
    Accuracy on testing data: 0.8718
    F-score on testing data: 0.7545
    
    Final Model trained on reduced data
    ------
    Accuracy on testing data: 0.8427
    F-score on testing data: 0.6997
    

### Question 8 - Effects of Feature Selection


- As expected, applying the model to a reduced dataset produces in a small decrease of accuracy (0.6997 instead of 0.7545) and f-score (0.8427 instead of 0.8718). 

- Nevertheless, the results are generally quite consistent that is why it could be relevant to training the model on the reduced data in the case of training time (GBC is the slowest of the 3 models choosen as we saw it earlier) is a significant factor. However, let us keep in mind that this decision would depend on the importance givent to accuracy and f-score in the evaluation of the model
