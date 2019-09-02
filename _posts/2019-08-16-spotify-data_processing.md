---
layout: post
title:  Spotify Music Analysis - Part 2
comments: true
tags: API spotify spotipy python data-mining
---


After a first post on data acquisition via the Spotify API, let's move on to the crucial step of each data science project: data processing and cleaning. 

### Objective of the notebook

Our goal is to process all the useful information that was collected in the previous notebook. Details of the operations performed will be provided as they occur in the notebook.


```python
#Import packages
import datetime
import pandas as pd
import pickle
import re
```


```python
#Load dataframes that we save earlier
loved_dataframe = pd.read_pickle("data/loved_dataframe.pkl")
hated_dataframe = pd.read_pickle("data/hated_dataframe.pkl")
```

## 1 - Data processing and feature engineering 

Before we merge our two dataframes into one, we want to identify the songs we like and the ones we don't so the first thing to do is to add a new column to differentiate the target variable.


```python
loved_dataframe['target'] = 1
hated_dataframe['target'] = 0
```


```python
final_df = pd.concat([loved_dataframe, hated_dataframe], axis=0)
print(final_df.shape)
```

    (1000, 34)
    


```python
final_df.sort_values(by='track_date', inplace=True)
final_df.reset_index(drop=True, inplace=True)
```

After a first inconclusive attempt to extract the musical genres from the list, it seemed to me to put aside the variable for a future treatment. For now, we are trying to retrieve the tags corresponding to the artist's number of subscribers (from a dictionary)


```python
final_df['artist_followers'] = final_df['artist_followers'].\
                                      astype(str).\
                                      apply(lambda x: re.search(r'\d+', x).group())
```

Conversion of features into the right format to facilitate the following analysis 


```python
final_df['artist_followers'] = final_df.artist_followers.astype('int')
```


```python
final_df['target_string'] = final_df['target'].astype(str) 
```


```python
final_df['correcteness_value'] = final_df['track_correctness'].astype(int) 
```

Creation of a new variable to determine the song's age, in this case the number of years since its release


```python
final_df['track_date'] = pd.to_datetime(final_df['track_date'])
final_df['track_year'] = final_df['track_date'].dt.year
final_df['track_age'] = int(datetime.datetime.now().year) - final_df['track_year']
```

We filter the dataframe by keeping only the necessary columns for our next step : the exploratory data analysis


```python
final_df = final_df.drop(columns=['track_type', 'track_uri',
                      'track_track_href', 'track_analysis_url',
                      'track_id','artist_id',
                      'artist_external_urls',
                      'artist_href', 'artist_images',
                      'artist_type','artist_uri'])
```


```python
final_df.shape
```




    (1000, 27)




```python
#Saving the processed dataframe 
final_df.to_pickle("data/final_dataframe.pkl")
```
