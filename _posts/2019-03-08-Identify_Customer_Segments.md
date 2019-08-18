---
layout: post
title:  "Udacity Data Science Nanodegree : Identify Customer Segments"
comments: true
tags: udacity python unsupervised-learning machine-learning
---


In this project, we will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that we will use has been provided by Bertelsmann Arvato Analytics, and represents a real-life data science task.


```python
# import libraries here; add more as necessary
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler

# magic word for producing visualizations in notebook
%matplotlib inline
```

### Step 0: Load the Data

There are four files associated with this project (not including this one):

- `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
- `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
- `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
- `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. we will use this information to cluster the general population into groups with similar demographic properties. Then, we will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.


```python
# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep = ';')

# Load in the feature summary file.
features = pd.read_csv('AZDIAS_Feature_Summary.csv', sep = ';')
```


```python
# Check the structure of the data after it's loaded (e.g. print the number of rows and columns, print the first few rows).

#Demographics dataset 
print('Number of columns', azdias.shape[1])
print('Number of rows:', azdias.shape[0])

#Features dataset
print('Number of columns', features.shape[1])
print('Number of rows:', features.shape[0])
```

    Number of columns 85
    Number of rows: 891221
    Number of columns 4
    Number of rows: 85


## Step 1: Preprocessing

### Step 1.1: Assess Missing Data

#### Step 1.1.1: Convert Missing Value Codes to NaNs


```python
# Identify missing or unknown data values and convert them to NaNs.
```


```python
# 1 - Check if there are missig values 
null_azdias = azdias.columns[azdias.isnull().any()]
azdias[null_azdias].isnull().sum()
```




    CJT_GESAMTTYP            4854
    GFK_URLAUBERTYP          4854
    LP_LEBENSPHASE_FEIN      4854
    LP_LEBENSPHASE_GROB      4854
    LP_FAMILIE_FEIN          4854
    LP_FAMILIE_GROB          4854
    LP_STATUS_FEIN           4854
    LP_STATUS_GROB           4854
    RETOURTYP_BK_S           4854
    SOHO_KZ                 73499
    TITEL_KZ                73499
    ALTER_HH                73499
    ANZ_PERSONEN            73499
    ANZ_TITEL               73499
    HH_EINKOMMEN_SCORE      18348
    KK_KUNDENTYP           584612
    W_KEIT_KIND_HH         107602
    WOHNDAUER_2008          73499
    ANZ_HAUSHALTE_AKTIV     93148
    ANZ_HH_TITEL            97008
    GEBAEUDETYP             93148
    KONSUMNAEHE             73969
    MIN_GEBAEUDEJAHR        93148
    OST_WEST_KZ             93148
    WOHNLAGE                93148
    CAMEO_DEUG_2015         98979
    CAMEO_DEU_2015          98979
    CAMEO_INTL_2015         98979
    KBA05_ANTG1            133324
    KBA05_ANTG2            133324
    KBA05_ANTG3            133324
    KBA05_ANTG4            133324
    KBA05_BAUMAX           133324
    KBA05_GBZ              133324
    BALLRAUM                93740
    EWDICHTE                93740
    INNENSTADT              93740
    GEBAEUDETYP_RASTER      93155
    KKK                    121196
    MOBI_REGIO             133324
    ONLINE_AFFINITAET        4854
    REGIOTYP               121196
    KBA13_ANZAHL_PKW       105800
    PLZ8_ANTG1             116515
    PLZ8_ANTG2             116515
    PLZ8_ANTG3             116515
    PLZ8_ANTG4             116515
    PLZ8_BAUMAX            116515
    PLZ8_HHZ               116515
    PLZ8_GBZ               116515
    ARBEIT                  97216
    ORTSGR_KLS9             97216
    RELAT_AB                97216
    dtype: int64



> Let's take a look at the column missing_or_unknown from the features dataset to check the different types of encoding for missing values


```python
features.missing_or_unknown.value_counts()
```




    [-1]        26
    [-1,9]      17
    [-1,0]      16
    [0]         12
    []          10
    [-1,X]       1
    [-1,0,9]     1
    [-1,XX]      1
    [XX]         1
    Name: missing_or_unknown, dtype: int64




```python
#Create a copy of features dataframe before transformation
features_raw = features.copy()
```

> After trying several methods (including the creation of several columns to store the values of the missing_or_unknown list and to use a dictionary of the said values to replace the matches by NaN), I chose a very simple, efficient and time-savingless solution


```python
#Change index of features dataset
features.set_index('attribute', inplace=True)
```


```python
#Function to convert rows of features dataset in integer whenever possible
def convert_int(value): 
    try:
        return int(value)
    except:
        return value
```


```python
#Create a dataframe before the replacement to be able to use the data later on
azdias_raw = azdias.copy()
```


```python
#Number of missing values in the original dataset
raw_missing_values = azdias_raw.isnull().sum()
raw_missing_values.sum()
```




    4896838




```python
#Convert matching missing values - after parsing missing_or_unknown column - between azdias and features in NaNs.
for col_name in azdias.columns:
    values_to_replace = features.loc[col_name]['missing_or_unknown'].strip('][').split(',')
    azdias[col_name] = azdias[col_name].replace([convert_int(x) for x in values_to_replace], np.nan)
```


```python
#Create a copy of the dataframe after replacement
clean_missing_values = azdias.copy()
```


```python
#Number of missing values after transformation
replacement = clean_missing_values.isnull().sum()
replacement.sum()
```




    8373929



#### Step 1.1.2: Assess Missing Data in Each Column


```python
# Perform an assessment of how much missing data there is in each column of the
# dataset.
missing_azdias_count = azdias.isnull().sum()
missing_azdias = np.round(missing_azdias_count[missing_azdias_count > 0] / len(azdias) * 100,2)
missing_azdias.sort_values(inplace=True, ascending=True)
```


```python
missing_azdias.plot.bar(figsize=(15,10),fontsize=13,color="indianred")
plt.xlabel('Columns with missing values')
plt.ylabel('Percentage of missing values')
plt.show();
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_23_0.png"  width="702" height="592">




```python
#Summary of missing data
missing_azdias.describe()
```




    count    61.000000
    mean     15.403115
    std      17.628423
    min       0.320000
    25%      10.450000
    50%      11.150000
    75%      13.070000
    max      99.760000
    dtype: float64



> On average, the columns contain more than 15% missing values. Let's take a look at the columns whose missing values rate is higher


```python
missing_azdias[missing_azdias > 20]
```




    ALTER_HH        34.81
    GEBURTSJAHR     44.02
    KBA05_BAUMAX    53.47
    KK_KUNDENTYP    65.60
    AGER_TYP        76.96
    TITEL_KZ        99.76
    dtype: float64



> By inspecting the variables with the most missing values - with the help of the data dictionary - it is clear that most of the information concerns categorical variables representing more or less sensitive personal data (age, date of birth of the head of the family, recent consumption behaviour). It is easy to understand why these attributes were not transmitted and/or anonymized for the purposes of the exercise. 


```python
#Detail of the missing values added before and after 
```


```python
#Number of missing values added after the transformation 
(replacement - raw_missing_values).sum()
```




    3477091




```python
#Compute the number of missing values before and after converting to NaNs
```


```python
missing_before = azdias_raw.isnull().sum()
missing_before = np.round(missing_before[missing_before > 0] / len(azdias_raw) * 100,2)
missing_before.sort_values(inplace=True, ascending=True)
```


```python
missing_after = clean_missing_values.isnull().sum()
missing_after = np.round(missing_after[missing_after > 0] / len(clean_missing_values) * 100,2)
missing_after.sort_values(inplace=True, ascending=True)
```


```python
plt.subplot(1, 2, 1)
missing_before.plot.bar(x='attribute', y='percent_missing', figsize=(18,10), grid=True, color="indianred")
plt.title(f'Missing values before transformation : {len(missing_before)} columns with missing values')
plt.ylabel('Percentage of missing values')

plt.subplot(1, 2, 2)
missing_after.plot.bar(x='attribute', y='percent_missing', grid=True, color="dodgerblue")
plt.title(f'Missing values after transformation : {len(missing_after)} columns with missing values')
plt.ylabel('Percentage of missing values')
plt.tight_layout()
plt.show();
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_33_0.png"  width="708" height="391">




```python
# Remove the outlier columns from the dataset
```


```python
missing_azdias_count.describe()
```




    count        85.000000
    mean      98516.811765
    std      146604.203317
    min           0.000000
    25%           0.000000
    50%       93148.000000
    75%      116515.000000
    max      889061.000000
    dtype: float64




```python
missing_azdias_count.hist(grid=False, color = 'indianred');
```


![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_36_0.png)



```python
missing_azdias_count[missing_azdias_count > 200000] 
```




    AGER_TYP        685843
    GEBURTSJAHR     392318
    TITEL_KZ        889061
    ALTER_HH        310267
    KK_KUNDENTYP    584612
    KBA05_BAUMAX    476524
    dtype: int64



> Looking at the statistics, we can see that only one column concentrates 889,061 missing values, about 99% of the total of the original dataset. However, the histogram of the number of missing values shows that the majority of the columns contain less than 200,000 missing values. It is this number that will serve as a threshold to eliminate outliers. Not surprisingly, the six columns - out of 85 - mentioned above are the ones with the most missing values and will therefore be eliminated. 


```python
#Remove selected columns
azdias_filtered = azdias.drop(['AGER_TYP','GEBURTSJAHR','TITEL_KZ','ALTER_HH','KK_KUNDENTYP','KBA05_BAUMAX'],axis=1)
```


```python
azdias_filtered.shape
```




    (891221, 79)




```python
#Number of missing values after removing outliers
missing_without_outliers_count = azdias_filtered.isnull().sum()
missing_without_outliers_count.sum()
```




    5035304




```python
missing_without_outliers = np.round(missing_without_outliers_count[missing_without_outliers_count > 0] 
                                    / len(azdias_filtered) * 100,2)
missing_without_outliers.sort_values(inplace=True, ascending=True)
```


```python
missing_without_outliers.plot.bar(figsize=(15,10),fontsize=13,color="indigo")
plt.xlabel('Columns with missing values')
plt.ylabel('Percentage of missing values')
plt.show();
```


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_43_0.png"  width="706" height="592">

#### Discussion 1.1.2: Assess Missing Data in Each Column


```python
# Investigate patterns in the amount of missing data in each column
```

> Of the 79 columns previously selected, only 55 columns contains outliers. Some groups stand out because they contain the same number of missing values (see the details below)


```python
#Converting to dataframe
without_outliers_df = pd.DataFrame({'columns':missing_without_outliers_count.index, 
                                    'count':missing_without_outliers_count.values})
```


```python
#Grouping columns by number of missing values
without_outliers_agg = [x for _, x in without_outliers_df.groupby(['count'])]
without_outliers_agg
```




    [                  columns  count
     1               ANREDE_KZ      0
     3       FINANZ_MINIMALIST      0
     4           FINANZ_SPARER      0
     5        FINANZ_VORSORGER      0
     6          FINANZ_ANLEGER      0
     7   FINANZ_UNAUFFAELLIGER      0
     8        FINANZ_HAUSBAUER      0
     9               FINANZTYP      0
     11       GREEN_AVANTGARDE      0
     22              SEMIO_SOZ      0
     23              SEMIO_FAM      0
     24              SEMIO_REL      0
     25              SEMIO_MAT      0
     26             SEMIO_VERT      0
     27             SEMIO_LUST      0
     28              SEMIO_ERL      0
     29             SEMIO_KULT      0
     30              SEMIO_RAT      0
     31             SEMIO_KRIT      0
     32              SEMIO_DOM      0
     33             SEMIO_KAEM      0
     34          SEMIO_PFLICHT      0
     35            SEMIO_TRADV      0
     39               ZABEOTYP      0,                 columns  count
     0  ALTERSKATEGORIE_GROB   2881,               columns  count
     2       CJT_GESAMTTYP   4854
     10    GFK_URLAUBERTYP   4854
     17     LP_STATUS_FEIN   4854
     18     LP_STATUS_GROB   4854
     21     RETOURTYP_BK_S   4854
     66  ONLINE_AFFINITAET   4854,                columns  count
     42  HH_EINKOMMEN_SCORE  18348,            columns  count
     37         SOHO_KZ  73499
     40    ANZ_PERSONEN  73499
     41       ANZ_TITEL  73499
     44  WOHNDAUER_2008  73499,         columns  count
     48  KONSUMNAEHE  73969,             columns  count
     15  LP_FAMILIE_FEIN  77792
     16  LP_FAMILIE_GROB  77792,              columns  count
     47       GEBAEUDETYP  93148
     49  MIN_GEBAEUDEJAHR  93148
     50       OST_WEST_KZ  93148
     51          WOHNLAGE  93148,                columns  count
     63  GEBAEUDETYP_RASTER  93155,        columns  count
     60    BALLRAUM  93740
     61    EWDICHTE  93740
     62  INNENSTADT  93740,                 columns  count
     14  LP_LEBENSPHASE_GROB  94572,          columns  count
     46  ANZ_HH_TITEL  97008,         columns  count
     77  ORTSGR_KLS9  97274,      columns  count
     76    ARBEIT  97375
     78  RELAT_AB  97375,                 columns  count
     13  LP_LEBENSPHASE_FEIN  97632,             columns  count
     52  CAMEO_DEUG_2015  99352
     53   CAMEO_DEU_2015  99352
     54  CAMEO_INTL_2015  99352,                 columns  count
     45  ANZ_HAUSHALTE_AKTIV  99611,              columns   count
     68  KBA13_ANZAHL_PKW  105800,                   columns   count
     20  PRAEGENDE_JUGENDJAHRE  108164,              columns   count
     19  NATIONALITAET_KZ  108315,         columns   count
     12   HEALTH_TYP  111196
     36  SHOPPER_TYP  111196
     38     VERS_TYP  111196,         columns   count
     69   PLZ8_ANTG1  116515
     70   PLZ8_ANTG2  116515
     71   PLZ8_ANTG3  116515
     72   PLZ8_ANTG4  116515
     73  PLZ8_BAUMAX  116515
     74     PLZ8_HHZ  116515
     75     PLZ8_GBZ  116515,         columns   count
     55  KBA05_ANTG1  133324
     56  KBA05_ANTG2  133324
     57  KBA05_ANTG3  133324
     58  KBA05_ANTG4  133324
     59    KBA05_GBZ  133324
     65   MOBI_REGIO  133324,            columns   count
     43  W_KEIT_KIND_HH  147988,      columns   count
     64       KKK  158064
     67  REGIOTYP  158064]




```python
col_group_missing_values = np.unique(without_outliers_df[ 'count'])
```


```python
print(f'{len(col_group_missing_values)} columns with the same number of missing values')
```

    25 columns with the same number of missing values
    

#### Step 1.1.3: Assess Missing Data in Each Row


```python
# How much data is missing in each row of the dataset?
missing_rows_values = azdias_filtered.isnull().sum(axis=1)
missing_rows_values
```




    0         43
    1          0
    2          0
    3          7
    4          0
    5          0
    6          0
    7          0
    8          0
    9          0
    10         0
    11        47
    12         6
    13         8
    14        47
    15         8
    16         6
    17        47
    18         3
    19         0
    20        10
    21         0
    22         0
    23         8
    24        47
    25         5
    26        19
    27         0
    28         0
    29         2
              ..
    891191     0
    891192     0
    891193     0
    891194     0
    891195     0
    891196     0
    891197     0
    891198     0
    891199     0
    891200     0
    891201     0
    891202     0
    891203    14
    891204     0
    891205     0
    891206     0
    891207     0
    891208     3
    891209     0
    891210     0
    891211     0
    891212     0
    891213     0
    891214     0
    891215     0
    891216     3
    891217     4
    891218     5
    891219     0
    891220     0
    Length: 891221, dtype: int64




```python
#Missing rows summary
missing_rows_values.describe()
```




    count    891221.000000
    mean          5.649894
    std          13.234687
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           3.000000
    max          49.000000
    dtype: float64




```python
#Plot the summary
plt.figure(figsize=(13,8))
missing_rows_values.hist()
plt.xlabel('Count of missing values')
plt.ylabel('Number of rows')
plt.show();
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_54_0.png"  width="705" height="425">



```python
# Divide the data into two subsets based on the number of missing
# values in each row (30 is the chosen threshold)
missing_values_high = missing_rows_values[missing_rows_values >=30]
missing_values_low = missing_rows_values[missing_rows_values < 30]
```


```python
# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
```


```python
#Function to compare distribution for subsets with few and several missing values
def compare_plots(column):
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Low missing values')
    sns.countplot(azdias_filtered.loc[missing_values_low.index,column])

    ax2 = fig.add_subplot(122)
    ax2.title.set_text('High missing values')
    sns.countplot(azdias_filtered.loc[azdias_filtered.index.isin(missing_values_high.index),column]);

    fig.suptitle(column)
    plt.show()
```


```python
#Create a list containing the columns to compare
col_names = ['ALTERSKATEGORIE_GROB', 'ANREDE_KZ', 'GFK_URLAUBERTYP', 'REGIOTYP', 'KKK']
```


```python
for col in col_names:
    compare_plots(col)
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_59_0.png"  width="703" height="414">


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_59_1.png"  width="703" height="414">


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_59_2.png"  width="703" height="414">


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_59_3.png"  width="703" height="414">


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_59_4.png"  width="703" height="414">


#### Discussion 1.1.3: Assess Missing Data in Each Row

> After removing the columns containing the most outliers, I became interested in the distribution of features with a lot of missing values ( more than 30) and those with none or few (less than 30). Although the majority of distributions are equivalent from one dataset to another, there are clear differences for three columns in particular, namely GFK_URLAUBERTYP, REGIOTYP and KKK.These variables correspond to socio-demographic data, namely the social class among the neighbourhood, the purchasing power of the individual at that of the average household at the local level and personal travel habits. Very discriminating information which, if isolated, can lead to biases in the way the data are analysed. That is why I think we should keep these variables and think about a way to manage the missing data without distorting the datatset.

### Step 1.2: Select and Re-Encode Features


```python
# How many features are there of each data type?
features_raw['type'].value_counts()
```




    ordinal        49
    categorical    21
    numeric         7
    mixed           7
    interval        1
    Name: type, dtype: int64



#### Step 1.2.1: Re-Encode Categorical Features


```python
# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
cat_var = features_raw[features_raw['type'] == 'categorical']['attribute'].values
```


```python
#We only keep the columns also present in our filtered dataset
cat_columns = [i for i in cat_var if i in azdias_filtered.columns]
```


```python
binary = []
multilevel = []
for col in cat_columns:
    if azdias_filtered[col].nunique()==2:
        binary.append(col)
    else:
        multilevel.append(col)

print(f'{len(binary)} binary columns : {(binary)}')
print(f'{len(multilevel)} multilevel columns : {(multilevel)}')
```

    5 binary columns : ['ANREDE_KZ', 'GREEN_AVANTGARDE', 'SOHO_KZ', 'VERS_TYP', 'OST_WEST_KZ']
    13 multilevel columns : ['CJT_GESAMTTYP', 'FINANZTYP', 'GFK_URLAUBERTYP', 'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN', 'LP_STATUS_GROB', 'NATIONALITAET_KZ', 'SHOPPER_TYP', 'ZABEOTYP', 'GEBAEUDETYP', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015']
    


```python
#Analyze binary columns
```


```python
azdias_filtered.ANREDE_KZ.value_counts()
```




    2    465305
    1    425916
    Name: ANREDE_KZ, dtype: int64




```python
azdias_filtered.GREEN_AVANTGARDE.value_counts()
```




    0    715996
    1    175225
    Name: GREEN_AVANTGARDE, dtype: int64




```python
azdias_filtered.SOHO_KZ.value_counts()
```




    0.0    810834
    1.0      6888
    Name: SOHO_KZ, dtype: int64




```python
azdias_filtered.VERS_TYP.value_counts()
```




    2.0    398722
    1.0    381303
    Name: VERS_TYP, dtype: int64




```python
azdias_filtered.OST_WEST_KZ.value_counts()
```




    W    629528
    O    168545
    Name: OST_WEST_KZ, dtype: int64




```python
# Re-encode categorical variable(s) to be kept in the analysis
```


```python
#Encoding binary columns 
azdias_filtered['OST_WEST_KZ'].replace(['W','O'], [1,0], inplace=True)
azdias_filtered['VERS_TYP'].replace([2.0,1.0], [1,0], inplace=True)
```


```python
#Analyze multilevel columns
```


```python
for col in multilevel:
    print(col)
    print(len(azdias_filtered[col].value_counts()))
```

    CJT_GESAMTTYP
    6
    FINANZTYP
    6
    GFK_URLAUBERTYP
    12
    LP_FAMILIE_FEIN
    11
    LP_FAMILIE_GROB
    5
    LP_STATUS_FEIN
    10
    LP_STATUS_GROB
    5
    NATIONALITAET_KZ
    3
    SHOPPER_TYP
    4
    ZABEOTYP
    6
    GEBAEUDETYP
    7
    CAMEO_DEUG_2015
    9
    CAMEO_DEU_2015
    44
    

> Removing unnecessary columns (too many categories, information already present in other features, irrelevant feature) to simplify the work prior to the Principal Component Analysis





```python
drop_multilevel = ['LP_STATUS_FEIN', 'CAMEO_DEU_2015', 'SHOPPER_TYP', 
                   'ZABEOTYP', 'LP_FAMILIE_FEIN', 'GEBAEUDETYP', 'FINANZTYP']
```


```python
drop_binary = ['SOHO_KZ','OST_WEST_KZ']
```


```python
drop_features = drop_multilevel + drop_binary
```


```python
drop_features
```




    ['LP_STATUS_FEIN',
     'CAMEO_DEU_2015',
     'SHOPPER_TYP',
     'ZABEOTYP',
     'LP_FAMILIE_FEIN',
     'GEBAEUDETYP',
     'FINANZTYP',
     'SOHO_KZ',
     'OST_WEST_KZ']




```python
azdias_encoded = azdias_filtered.drop(drop_features,axis=1)
```


```python
azdias_encoded.shape
```




    (891221, 70)




```python
#Filter columns to keep before one hot encoding
```


```python
cols_to_keep = []
for col in multilevel:
    if col not in drop_multilevel:
        cols_to_keep.append(col)

for col in binary:
    if col not in drop_binary:
        cols_to_keep.append(col)
```


```python
cols_to_keep
```




    ['CJT_GESAMTTYP',
     'GFK_URLAUBERTYP',
     'LP_FAMILIE_GROB',
     'LP_STATUS_GROB',
     'NATIONALITAET_KZ',
     'CAMEO_DEUG_2015',
     'ANREDE_KZ',
     'GREEN_AVANTGARDE',
     'VERS_TYP']




```python
azdias_encoded = pd.get_dummies(azdias_encoded, columns=cols_to_keep)
```





```python
azdias_encoded.shape
```




    (891221, 107)



#### Discussion 1.2.1: Re-Encode Categorical Features

> After separating the columns by type to analyze the number of values represented, there are significant differences between the variables, as some contain only two unique values while others have more than about 40 distinct values. However, this does not mean that this level of detail is not interesting, but it is a question of whether this level of granularity is interesting on the one hand and useful on the other hand with regard to the type of information it represents. 
>> I therefore chose to exclude some variables for two reasons, namely :
- a too large number of categories and irrelevant in relation to the information conveyed 
- duplicates with existing columns, keeping large scale for socio-economic characteristics

#### Step 1.2.2: Engineer Mixed-Type Features

There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention :
- "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, we should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
- "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).


```python
#Analyze mixed type columns
mix_var = features_raw[features_raw['type'] == 'mixed']['attribute'].values
mix_columns = [i for i in mix_var if i in azdias_filtered.columns]
print(mix_columns)
```

    ['LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015', 'PLZ8_BAUMAX']
    


```python
# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
azdias_encoded['PRAEGENDE_JUGENDJAHRE'].value_counts()
```




    14.0    188697
    8.0     145988
    5.0      86416
    10.0     85808
    3.0      55195
    15.0     42547
    11.0     35752
    9.0      33570
    6.0      25652
    12.0     24446
    1.0      21282
    4.0      20451
    2.0       7479
    13.0      5764
    7.0       4010
    Name: PRAEGENDE_JUGENDJAHRE, dtype: int64




```python
#Engineer generation and movement variables based on PRAEGENDE_JUGENDJAHRE
```


```python
#Create dictionaries for new values
generation_dict = {1:1,2:1,3:2,4:2,5:3,6:3,7:3,8:4,9:4,10:5,11:5,12:5,13:5,14:6,15:6}
movement_dict = {1:0, 2:1, 3:0, 4:1, 5:0, 6:1, 7:1, 8:0, 9:1, 10:0, 11:1, 12:0, 13:1, 14:0, 15:1}
```


```python
#Map values from dict to dataframe
azdias_encoded['JUGENDJAHRE_DECADE'] = azdias_encoded['PRAEGENDE_JUGENDJAHRE'].map(generation_dict)
azdias_encoded['JUGENDJAHRE_MOVEMENT'] = azdias_encoded['PRAEGENDE_JUGENDJAHRE'].map(movement_dict)
```



```python
# Investigate "CAMEO_INTL_2015" and engineer two new variables
```


```python
azdias_encoded['CAMEO_INTL_2015'].value_counts()
```




    51    133694
    41     92336
    24     91158
    14     62884
    43     56672
    54     45391
    25     39628
    22     33155
    23     26750
    13     26336
    45     26132
    55     23955
    52     20542
    31     19024
    34     18524
    15     16974
    44     14820
    12     13249
    35     10356
    32     10354
    33      9935
    Name: CAMEO_INTL_2015, dtype: int64




```python
#Create dictionaries for new values
wealth_dict = {'11':1,'12':1,'13':1,'14':1,'15':1,
               '21':2,'22':2,'23':2,'24':2,'25':2,
               '31':3,'32':3,'33':3,'34':3,'35':3,
               '41':4,'42':4,'43':4,'44':4,'45':4,
               '51':5,'52':5,'53':5,'54':5,'55':5}

lifestage_dict = {'11':1,'12':2,'13':3,'14':4,'15':5,
                  '21':1,'22':2,'23':3,'24':4,'25':5,
                  '31':1,'32':2,'33':3,'34':4,'35':5,
                  '41':1,'42':2,'43':3,'44':4,'45':5,
                  '51':1,'52':2,'53':3,'54':4,'55':5}          
```


```python
#Map values from dict to dataframe
azdias_encoded['CAMEO_WEALTH'] = azdias_encoded['CAMEO_INTL_2015'].map(wealth_dict)
azdias_encoded['CAMEO_LIFESTAGE'] = azdias_encoded['CAMEO_INTL_2015'].map(lifestage_dict)
```

#### Discussion 1.2.2: Engineer Mixed-Type Features

> The purpose of this transformation is to separate the informations contained in the columns 
PRAEGENDE_JUGENDJAHRE and CAMEO_INTL_2015 to end up with an information by variable : 
- generation and movement for PRAEGENDE_JUGENDJAHRE 
- wealth and life stage for the column CAMEO_INTL_2015

#### Step 1.2.3: Complete Feature Selection


```python
#Filter WOHNLAGE to create rural and quality of neighborhood new features
azdias_encoded['RURAL_NEIGHBORHOOD'] = azdias_encoded['WOHNLAGE'].map({0:0,1:0,2:0,3:0,4:0,5:0,7:1,8:1})
azdias_encoded['NEIGHBORHOOD_RANK'] = azdias_encoded['WOHNLAGE'].map({0:0,1:1,2:2,3:3,4:4,5:5,7:0,8:0})
```




```python
cols_to_drop = ['PRAEGENDE_JUGENDJAHRE',
                'LP_LEBENSPHASE_FEIN',
                'CAMEO_INTL_2015',
                'WOHNLAGE',
                'PLZ8_BAUMAX']
```


```python
#Drop unuseful columns
azdias_final = azdias_encoded.drop(cols_to_drop,axis=1)
```





```python
azdias_final.shape
```




    (891221, 108)




```python
# Ensure that the dataframe only contains the columns that should be passed to the algorithm functions
```

**# Step 1** 
> Same operation for the WOHNLAGE column that is filtered to create two new variables: 
- the quality of life in a neighbourhood 
- whether or not this neighbourhood is urban or rural on the other hand 

**# Step 2** 
> We remove the variables that are useless to our analysis, such as :
- the variables from which variables (news columns) have been created previously (PRAEGENDE_JUGENDJAHRE, CAMEO_INTL_2015, WOHNLAGE)
- variables whose information is contained in other columns (LP_LEBENSPHASE_FEIN, PLZ8_BAUMAX)

### Step 1.3: Create a Cleaning Function


```python
def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    #Loading data
    features_raw = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')
    
    #Create a copy of features dataset to apply future transformations
    features = features_raw.copy()
    features.set_index('attribute', inplace=True)
    
    #Convert missing values to NaNs 
    def convert_int(value): 
        try:
            return int(value)
        except:
            return value

    for col_name in df.columns:
        values_to_replace = features.loc[col_name]['missing_or_unknown'].strip('][').split(',')
        df[col_name] = df[col_name].replace([convert_int(x) for x in values_to_replace], np.nan)
    
    #Assess missing data for rows and columns and removing outliers 
    missing_df_count = df.isnull().sum()
    missing_df = np.round(missing_df_count[missing_df_count > 0] / len(df) * 100,2).sort_values(ascending=True)
    df_filtered = df.drop(['AGER_TYP','GEBURTSJAHR','TITEL_KZ','ALTER_HH','KK_KUNDENTYP','KBA05_BAUMAX'], axis=1)
    missing_rows_df = df_filtered.isnull().sum(axis=1)
    
    #Re-encode, engineer and select columns
    cat_var = features_raw[features_raw['type'] == 'categorical']['attribute'].values
    cat_columns = [i for i in cat_var if i in df_filtered.columns]

    binary = []
    multilevel = []
    for col in cat_columns:
        if df_filtered[col].nunique()>2:
            multilevel.append(col)
    else:
        binary.append(col)
        
    df_filtered['OST_WEST_KZ'].replace(['W','O'], [1,0], inplace=True)
    df_filtered['VERS_TYP'].replace([2.0,1.0], [1,0], inplace=True)
    
    drop_multilevel = ['LP_STATUS_FEIN', 'CAMEO_DEU_2015', 
                       'SHOPPER_TYP', 'ZABEOTYP', 
                       'LP_FAMILIE_FEIN', 'GEBAEUDETYP', 'FINANZTYP']
    
    drop_binary = ['SOHO_KZ', 'OST_WEST_KZ']
    drop_features = drop_multilevel + drop_binary
    
    df_encoded = df_filtered.drop(drop_features,axis=1)
    
    cols_to_keep = ['CJT_GESAMTTYP',
                    'GFK_URLAUBERTYP',
                    'LP_FAMILIE_GROB',
                    'LP_STATUS_GROB',
                    'NATIONALITAET_KZ',
                    'CAMEO_DEUG_2015',
                    'ANREDE_KZ',
                    'GREEN_AVANTGARDE',
                    'VERS_TYP']
    
    df_encoded = pd.get_dummies(df_encoded, columns=cols_to_keep)
    
    generation_dict = {1:1,2:1,3:2,4:2,5:3,6:3,7:3,8:4,9:4,10:5,11:5,12:5,13:5,14:6,15:6}
    movement_dict = {1:0, 2:1, 3:0, 4:1, 5:0, 6:1, 7:1, 8:0, 9:1, 10:0, 11:1, 12:0, 13:1, 14:0, 15:1}
    df_encoded['JUGENDJAHRE_DECADE'] = df_encoded['PRAEGENDE_JUGENDJAHRE'].map(generation_dict)
    df_encoded['JUGENDJAHRE_MOVEMENT'] = df_encoded['PRAEGENDE_JUGENDJAHRE'].map(movement_dict)
    
    wealth_dict = {'11':1,'12':1,'13':1,'14':1,'15':1,
                   '21':2,'22':2,'23':2,'24':2,'25':2,
                   '31':3,'32':3,'33':3,'34':3,'35':3,
                   '41':4,'42':4,'43':4,'44':4,'45':4,
                   '51':5,'52':5,'53':5,'54':5,'55':5}
    
    lifestage_dict = {'11':1,'12':2,'13':3,'14':4,'15':5,
                      '21':1,'22':2,'23':3,'24':4,'25':5,
                      '31':1,'32':2,'33':3,'34':4,'35':5,
                      '41':1,'42':2,'43':3,'44':4,'45':5,
                      '51':1,'52':2,'53':3,'54':4,'55':5}
    df_encoded['CAMEO_WEALTH'] = df_encoded['CAMEO_INTL_2015'].map(wealth_dict)
    df_encoded['CAMEO_LIFESTAGE'] = df_encoded['CAMEO_INTL_2015'].map(lifestage_dict)
    
    df_encoded['RURAL_NEIGHBORHOOD'] = df_encoded['WOHNLAGE'].map({0:0,1:0,2:0,3:0,4:0,5:0,7:1,8:1})
    df_encoded['NEIGHBORHOOD_RANK'] = df_encoded['WOHNLAGE'].map({0:0,1:1,2:2,3:3,4:4,5:5,7:0,8:0})
    
    cols_to_drop = ['PRAEGENDE_JUGENDJAHRE','LP_LEBENSPHASE_FEIN',
                    'CAMEO_INTL_2015','WOHNLAGE','PLZ8_BAUMAX']
    
    df_final = df_encoded.drop(cols_to_drop,axis=1)
    
    return df_final
```


```python
df_cleaned = clean_data(azdias)
```


```python
#We check that the shape of the dataframe is the same between the result of the function and the dataset that has been transformed step by step
set(azdias_final.columns).symmetric_difference(df_cleaned.columns)
```




    set()



## Step 2: Feature Transformation

### Step 2.1: Apply Feature Scaling




```python
#Assessment of missing values in the dataset
df_cleaned.isnull().sum()
```




    ALTERSKATEGORIE_GROB       2881
    FINANZ_MINIMALIST             0
    FINANZ_SPARER                 0
    FINANZ_VORSORGER              0
    FINANZ_ANLEGER                0
    FINANZ_UNAUFFAELLIGER         0
    FINANZ_HAUSBAUER              0
    HEALTH_TYP               111196
    LP_LEBENSPHASE_GROB       94572
    RETOURTYP_BK_S             4854
    SEMIO_SOZ                     0
    SEMIO_FAM                     0
    SEMIO_REL                     0
    SEMIO_MAT                     0
    SEMIO_VERT                    0
    SEMIO_LUST                    0
    SEMIO_ERL                     0
    SEMIO_KULT                    0
    SEMIO_RAT                     0
    SEMIO_KRIT                    0
    SEMIO_DOM                     0
    SEMIO_KAEM                    0
    SEMIO_PFLICHT                 0
    SEMIO_TRADV                   0
    ANZ_PERSONEN              73499
    ANZ_TITEL                 73499
    HH_EINKOMMEN_SCORE        18348
    W_KEIT_KIND_HH           147988
    WOHNDAUER_2008            73499
    ANZ_HAUSHALTE_AKTIV       99611
                              ...  
    LP_FAMILIE_GROB_5.0           0
    LP_STATUS_GROB_1.0            0
    LP_STATUS_GROB_2.0            0
    LP_STATUS_GROB_3.0            0
    LP_STATUS_GROB_4.0            0
    LP_STATUS_GROB_5.0            0
    NATIONALITAET_KZ_1.0          0
    NATIONALITAET_KZ_2.0          0
    NATIONALITAET_KZ_3.0          0
    CAMEO_DEUG_2015_1             0
    CAMEO_DEUG_2015_2             0
    CAMEO_DEUG_2015_3             0
    CAMEO_DEUG_2015_4             0
    CAMEO_DEUG_2015_5             0
    CAMEO_DEUG_2015_6             0
    CAMEO_DEUG_2015_7             0
    CAMEO_DEUG_2015_8             0
    CAMEO_DEUG_2015_9             0
    ANREDE_KZ_1                   0
    ANREDE_KZ_2                   0
    GREEN_AVANTGARDE_0            0
    GREEN_AVANTGARDE_1            0
    VERS_TYP_0.0                  0
    VERS_TYP_1.0                  0
    JUGENDJAHRE_DECADE       108164
    JUGENDJAHRE_MOVEMENT     108164
    CAMEO_WEALTH              99352
    CAMEO_LIFESTAGE           99352
    RURAL_NEIGHBORHOOD        93148
    NEIGHBORHOOD_RANK         93148
    Length: 108, dtype: int64




```python
#Extraction of columns with at least one missing value
columns_with_na = df_cleaned.columns[df_cleaned.isnull().any()].tolist()
```


```python
#Checking columns' type for variables with missing values
for colname, coltype in df_cleaned.dtypes.iteritems():
    if colname in columns_with_na:
        print(colname, coltype)
```

    ALTERSKATEGORIE_GROB float64
    HEALTH_TYP float64
    LP_LEBENSPHASE_GROB float64
    RETOURTYP_BK_S float64
    ANZ_PERSONEN float64
    ANZ_TITEL float64
    HH_EINKOMMEN_SCORE float64
    W_KEIT_KIND_HH float64
    WOHNDAUER_2008 float64
    ANZ_HAUSHALTE_AKTIV float64
    ANZ_HH_TITEL float64
    KONSUMNAEHE float64
    MIN_GEBAEUDEJAHR float64
    KBA05_ANTG1 float64
    KBA05_ANTG2 float64
    KBA05_ANTG3 float64
    KBA05_ANTG4 float64
    KBA05_GBZ float64
    BALLRAUM float64
    EWDICHTE float64
    INNENSTADT float64
    GEBAEUDETYP_RASTER float64
    KKK float64
    MOBI_REGIO float64
    ONLINE_AFFINITAET float64
    REGIOTYP float64
    KBA13_ANZAHL_PKW float64
    PLZ8_ANTG1 float64
    PLZ8_ANTG2 float64
    PLZ8_ANTG3 float64
    PLZ8_ANTG4 float64
    PLZ8_HHZ float64
    PLZ8_GBZ float64
    ARBEIT float64
    ORTSGR_KLS9 float64
    RELAT_AB float64
    JUGENDJAHRE_DECADE float64
    JUGENDJAHRE_MOVEMENT float64
    CAMEO_WEALTH float64
    CAMEO_LIFESTAGE float64
    RURAL_NEIGHBORHOOD float64
    NEIGHBORHOOD_RANK float64
    


```python
df_raw = df_cleaned.copy()
```


```python
#Before performing any operations, the names of the dataset variables are kept apart
header_names = list(df_raw.columns.values)
```


```python
imputer = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
df_processed = imputer.fit_transform(df_raw)
df_processed = pd.DataFrame(df_processed)
```






```python
# Apply feature scaling to the general population demographics data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_processed)
df_scaled = pd.DataFrame(df_scaled,columns=header_names)
```





### Discussion 2.1: Apply Feature Scaling

> Since variables containing missing values are mostly categorical and that the values were processed earlier (re-encoding step), it seemed appropriate to replace the missing values by the most frequent ones for the columns concerned. Replacing these with the median or mean was not the best solution considering the real nature of the data, as decimal numbers are only a reflection of categorical variables. As suggested, I used StandardScaler to normalize the values.

### Step 2.2: Perform Dimensionality Reduction



```python
# Apply PCA to the data
pca = PCA()
pca_azdias = pca.fit_transform(df_scaled)
```


```python
# Investigate the variance accounted for by each principal component
nb_of_components = len(pca.explained_variance_ratio_)
ind = np.arange(nb_of_components)
var_explained = pca.explained_variance_ratio_

plt.figure(figsize=(15,10))
cumsum_variances = np.cumsum(var_explained)
plt.plot(ind, cumsum_variances)
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='indianred')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') 
print('Cumulative variance explained with 60 components: {}'.format(cumsum_variances[61]))
plt.show();
```

    Cumulative variance explained with 60 components: 0.9266083646879966
    

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_136_1.png"  width="704" height="468">



```python
# Re-apply PCA to the data while selecting for number of components to retain.
pca_azdias = PCA(n_components=61, random_state=42)
azdias_pca = pca_azdias.fit_transform(df_scaled)
```


```python
pca_azdias.explained_variance_ratio_.sum()
```




    0.92267467657032987



### Discussion 2.2: Perform Dimensionality Reduction

> By looking at the plot, we notice that 60 components were sufficient to explain more than 90% of the data. I could increase that value but starting with that thresold seems sufficient to me

### Step 2.3: Interpret Principal Components


```python
def pca_analysis(dataset, pca):
    '''
    Create a dataframe to store the results of PCA
    Include dimension feature weights and explained variance
    '''

    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca_azdias.components_)+1)]

    components = pd.DataFrame(np.round(pca_azdias.components_, 4), columns = list(df_scaled.keys()))
    components.index = dimensions

    ratios = pca.explained_variance_ratio_.reshape(len(pca_azdias.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    
    return pd.merge(variance_ratios, components, left_index=True, right_index=True)
```


```python
results_pca = pca_analysis(df_scaled, pca_azdias)
```


```python
def plot_pca(data, pca, compo_num):
    #Vizualise best and worst values for each dimension
    compo = pd.DataFrame(np.round(pca_azdias.components_, 4), columns = data.keys()).iloc[compo_num-1]
    compo.sort_values(ascending=False, inplace=True)
    compo = pd.concat([compo.head(5), compo.tail(5)])
    fig, ax = plt.subplots(figsize = (15,10))
    compo.plot(ax = ax, kind = 'bar', title='Component' + ' ' + str(compo_num))
    ax.grid(linewidth='0.5', alpha=0.5)
    ax.set_axisbelow(True)
    plt.show();
```


```python
plot_pca(df_scaled, pca_azdias, 0)
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_145_0.png"  width="700" height="554">



```python
# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.

plot_pca(df_scaled, pca_azdias, 1)
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_146_0.png"  width="700" height="562">



```python
# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.
plot_pca(df_scaled, pca_azdias, 2)
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_147_0.png"  width="700" height="515">



```python
# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.
plot_pca(df_scaled, pca_azdias, 3)
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_148_0.png"  width="700" height="515">


### Discussion 2.3: Interpret Principal Components

Here are the results for the first four components

> **Component #0**

Positively correlated with :
- ANZ_HAUSHALTE_AKTIV : Number of households in the building
- SEMIO_SOZ : Personality typology, in this case socially-minded
- SEMIO_MAT : Personality typology, in this case materialistic
- SEMIO_KRIT : Personality typology, in this case critical-minded
- KBA05_GBZ : Number of buildings in the microcell

Negatively correlated with :
- KBA05_ANTG4 : Number of 10+ family houses in the microcell
- PLZ8_ANTG4 : Number of 10+ family houses in the PLZ8 region
- SEMIO_LUST : Personality typology, in this case sensual-minded
- ANZ_HH_TITEL : Number of professional academic title holders in building
- LP_STATUS_GROB : Social status, in this case houseowners 

*The principal component seems describe a mix of features related to economic situation of an area (wealth, density of population) and personality traits of individuals*

> **Component #1**

Positively correlated with :
- PLZ8_ANTG3 : Number of 6-10 family houses in the PLZ8 region
- CAMEO_WEALTH : Household's wealth (increasing order of poverty)
- LP_STATUS_GROB_1 : Social status, in this case low-income earneer 
- EWDICHTE : Density of households per square kilometer
- ORTSGR_KLS9 : Size of community

Negatively correlated with :
- KBA05_GBZ : Number of buildings in the microcell
- PLZ8_ANTG1 : Number of 1-2 family houses in the PLZ8 region
- FINANZ_MINIMALIST : Financial typology, in this case low financial interest
- KBA05_ANTG1 : Number of 1-2 family houses in the microcell
- MOBI_REGIO : Movement patterns

*The principal component seems describe a mix of features related to population density in a specific area and the financial condition in the area concerned*

> **Component #2**

Positively correlated with :
- ALTERSKATEGORIE_GROB : Estimated age based on given name analysis 
- FINANZ_VORSORGER : Financial typology, in this case be prepared
- SEMIO_ERL : Personality typology, in this case event-oriented
- SEMIO_LUST : Personality typology, in this case sensual-minded
- RETOURTYP_BK_S : Return type

Negatively correlated with :
- SEMIO_PFLICHT : Personality typology, in this case dutiful
- FINANZ_SPARER : Financial typology, in this case money-saver (higher means lower topology)
- FINANZ_UNAUFFAELLIGER: Financial typology, in this case inconspicuous
- SEMIO_REL : Personality typology, in this case religious
- JUGENDJAHRE_DECADE : Generation of an person according to his year of birth (in decades) 

*The principal component seems describe a mix of features related to personality traits and socio-demographic attributes of individuals*

> **Component #3**

Positively correlated with :
- SEMIO_VERT : Personality typology, in this case dreamful
- ANREDE_KZ_1 : Gender, in this case male
- SEMIO_SOZ : Personality typology, in this case socially-minded
- SEMIO_KULT : Personality typology, in this case cultural-minded
- SEMIO_FAM : Personality typology, family-minded

Negatively correlated with :
- SEMIO_ERL : Personality typology, in this case event-oriented
- SEMIO_KRIT : Personality typology, in this case critical-minded
- SEMIO_DOM : Personality typology, in this case dominant-minded
- SEMIO_KAEM : Personality typology, in this case combative attitude
- ANREDE_KZ_2 : Gender, in this case female

*The principal component seems describe a mix of features related to gender of individuals and personalities*

## Step 3: Clustering

### Step 3.1: Apply Clustering to General Population


```python
# Over a number of different cluster counts...
# run k-means clustering on the data and...
# compute the average within-cluster distances
```


```python
#Using a sample (20% of the dataset) to reduce computation time
azdias_pca_sample = azdias_pca[np.random.choice(azdias_pca.shape[0], \
                                                int(azdias_pca.shape[0]*0.2), replace=False)]
```


```python
scores = []
for x in list(range(2,48,2)):
    kmeans = MiniBatchKMeans(x)
    kmeans.fit(azdias_pca_sample)
    print(x,kmeans.score(azdias_pca_sample))
    scores.append(kmeans.score(azdias_pca_sample))
```

    2 -16621985.9032
    4 -14757274.8278
    6 -14066145.6935
    8 -13114682.717
    10 -13022089.973
    12 -12320265.0798
    14 -12183168.6079
    16 -12042757.0266
    18 -11993575.2186
    20 -11871896.4095
    22 -11683475.5874
    24 -11446900.2931
    26 -11519249.2355
    28 -11338999.7492
    30 -11226087.2016
    32 -11103023.1939
    34 -10979667.8265
    36 -11023039.1087
    38 -10874500.911
    40 -10875113.8665
    42 -10763970.7824
    44 -10689397.4284
    46 -10604883.6178
    


```python
# Investigate the change in within-cluster distance across number of clusters.
```


```python
scores_to_plot = [abs(i) for i in scores]
x = range(2,48,2)
plt.figure(figsize=(15, 10))
plt.plot(x, scores_to_plot, linestyle='--',linewidth=3, marker='o', color='indianred')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('SSE vs. Number of Clusters')
plt.show();
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_156_0.png"  width="700" height="475">



```python
# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
```


```python
start_time = time.time()

n_clusters = 24
kmeans = KMeans(n_clusters = n_clusters, random_state=42)
azdias_clusters = kmeans.fit_predict(azdias_pca);

print("--- Run time: %s mins ---" % np.round(((time.time() - start_time)/60),2))
```

    --- Run time: 11.82 mins ---
    

> I started by setting the number of clusters at 30. I have therefore decided to reduce this threshold to 24 on the one hand, to avoid memory problems and on the other hand, because there is no significant difference beyond this threshold. 

### Discussion 3.1: Apply Clustering to General Population

### Step 3.2: Apply All Steps to the Customer Data


```python
# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', delimiter=';')
```


```python
# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

customers_cleaned = clean_data(customers)
customers_cleaned.isnull().sum()

customers_na = customers_cleaned.columns[customers_cleaned.isnull().any()].tolist()
customers_names = list(customers_cleaned.columns.values)

imputer = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
customers_processed = imputer.fit_transform(customers_cleaned)
customers_processed = pd.DataFrame(customers_processed)

scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers_processed)
customers_scaled = pd.DataFrame(customers_scaled,columns=customers_names)
```


```python
pca_customers = PCA(n_components=61, random_state=42)
customers_pca = pca_customers.fit_transform(customers_scaled)
```


```python
customers_pca = pd.DataFrame(customers_pca)
```


```python
customer_clusters = kmeans.predict(customers_pca)
```

### Step 3.3: Compare Customer Data to Demographics Data


```python
# Converting predicted labels to dataframes
labels_azdias = pd.DataFrame(azdias_clusters, index=range(len(azdias_clusters)), columns=['Cluster'])
labels_customers = pd.DataFrame(customer_clusters, index=range(len(customer_clusters)), columns=['Cluster'])
```


```python
# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.
```


```python
n_population = []
n_customer = []
for i in range(n_clusters):
    n_population.append(np.argwhere(azdias_clusters==i).shape[0]/azdias_clusters.shape[0])
    n_customer.append(np.argwhere(customer_clusters==i).shape[0]/customer_clusters.shape[0])


label_counts = pd.DataFrame({'Global Dataset': n_population})
label_counts['Customer Dataset'] = pd.Series(n_customer)
label_counts['Clusters'] = pd.Series(list(range(n_clusters)))

# Plotting bar chart to visualize these proportions
customer_props = label_counts['Customer Dataset']
general_props = label_counts['Global Dataset']

ind = np.arange(n_clusters) 
width = 0.35 

plt.figure(figsize=(15,7))
plt.bar(ind, general_props, width, color='indigo', label='General')
plt.bar(ind + width, customer_props, width, color='gold',
    label='Customer')

plt.ylabel('Percentage')
plt.xlabel('Cluster')
plt.legend(loc='best')
plt.show();
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_170_0.png"  width="700" height="332">


### Discussion 3.3: Compare Customer Data to Demographics Data


```python
# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?
```


```python
over_represented_cluster = azdias_clusters==13
plt.bar(np.arange(20), np.mean(azdias_pca[over_represented_cluster][:,:20], axis=0))
```




    <Container object of 20 artists>




![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_173_1.png)


> We filter the data from the customer dataset corresponding to cluster 13 and then determine which main components are most represented (0,3,4) for the cluster. 
3 typical profiles stand out from the group :

- Young people (probably students) living in a residence mainly composed of studios or two rooms, open-minded, sociable and not very materialistic
- Single men, rather dreamy, sociable and interested in the arts
- Very wealthy, urban and traditionalist households, sensitive to ecology


```python
# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?
under_represented_cluster = azdias_clusters==5
plt.bar(np.arange(20), np.mean(azdias_pca[under_represented_cluster][:,:20], axis=0))
```




    <Container object of 20 artists>




![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/identify_customer_segments/output_175_1.png)


> Same as above, 4 typical profiles stand out from the group :
- Middle-class households, living on the suburbs and travelling mainly by car, traditional and risk averse
- Elderly people living in families, living in a chic and conservative neighbourhood, wealthy and with purchasing power in the high average
- Very wealthy, urban and traditionalist households, sensitive to ecology 
- Conservative low-income German households living close to local shops
