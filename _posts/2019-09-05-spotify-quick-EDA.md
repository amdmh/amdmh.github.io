---
layout: post
title:  "Spotify Music Analysis - Part 3 "
comments : true
tags: API spotify spotipy python data-analysis
---

```python
#Import packages
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import matplotlib.pyplot as plt
%matplotlib inline
```


```python
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 50)
pd.options.display.max_colwidth = 500
```


```python
#load dataframe of playlists previously saved
df = pd.read_pickle('final_dataframe.pkl')
```


```python
df.shape
```




    (1000, 27)




```python
#Checking for missing values
df.isnull().sum()
```




    track_danceability          0
    track_energy                0
    track_key                   0
    track_loudness              0
    track_mode                  0
    track_speechiness           0
    track_acousticness          0
    track_instrumentalness      0
    track_liveness              0
    track_valence               0
    track_tempo                 0
    track_duration_ms           0
    track_time_signature        0
    track_name                  0
    track_date                  0
    track_popularity            0
    track_preview             396
    track_correctness           0
    artist_followers            0
    artist_genres               0
    artist_name                 0
    artist_popularity           0
    target                      0
    target_string               0
    correcteness_value          0
    track_year                  0
    track_age                   0
    dtype: int64



**The idea of exploiting musical genres has for the moment been abandoned, due to the lack of relevance in the results obtained (columns encoded in variable dummies, with little readability). This makes it difficult to verify the relevance of the variable. However, it is not excluded that I may devote a post to an in-depth exploratory analysis aimed at characterizing my musical tastes and, in this case, the extracted data, whatever their form, will have their place.**

## Visualisation

To understand this better, let's visualize the distribution of the numerical variables to see if there are "significant" differences (not in a statistical sense but rather to the extent that they jump out at us) between the songs I like and those I don't like


```python
numeric_features = ['track_acousticness',
                    'track_danceability',
                    'track_duration_ms',
                    'track_energy',
                    'track_instrumentalness',
                    'track_liveness',
                    'track_loudness',
                    'track_speechiness',
                    'track_valence',
                    'track_tempo',
                    'track_time_signature',
                    'track_popularity',
                    'track_age',
                    'artist_popularity',
                    'artist_followers']
```


```python
#Create a function to manage positions of subplots
def create_list_of_positions(numeric_features):
    nrows_subplots = int(np.ceil((len(numeric_features) / 2)))
    a = list(np.repeat(np.arange(nrows_subplots)+1,2))
    b = list(np.tile(np.arange(2)+1,nrows_subplots))
    a = [int(items) for items in a]
    b = [int(items) for items in b]
    positions = list(zip(a,b))
    return positions
```


```python
subplots_positions = create_list_of_positions(numeric_features)
```


```python
#import chart_studio.plotly 
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=False)
#import plotly.graph_objs as go
#import plotly.tools as tls
#import plotly.offline as py#
```


```python
# Unable to display Plotly on Github Pages, the graph rendering is shown below - 
# Module used is orca allows to export plotly graphs as static images

#from plotly.subplots import make_subplots
#import plotly.graph_objects as go

# Initialize figure with subplots
#fig = make_subplots(
    #rows=max(subplots_positions,key=lambda x:x[0])[0], 
    #cols=max(subplots_positions,key=lambda x:x[1])[1],
    #subplot_titles=(numeric_features)
#)

#for position, f in enumerate(numeric_features):
    #fig.add_trace(go.Box(y=df[f].loc[df.target == 0], name='hated songs', 
                     #marker_color='red'),row=subplots_positions[position][0], col=subplots_positions[position][1])
    #fig.add_trace(go.Box(y=df[f].loc[df.target == 1], name='loved songs', 
                     #marker_color='blue'),row=subplots_positions[position][0], col=subplots_positions[position][1])
    

#fig.update_layout(title_text="Distributions of features : Liked vs. Hated Songs", 
                  #title_font_size=25, height=3000, showlegend=False)
#fig.show()
```


```python
from IPython.display import Image
Image("images/fig1.png")
```




![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/spotify/output_13_0.png)



At first glance, it seems that the distribution of the song's popularity, age and energy according to the target variable seem quite heterogeneous. Are these the most important variables to make the difference between what I like and where we like? It is too early to say. Next step : modelling ! 


```python
df.columns
```




    Index(['track_danceability', 'track_energy', 'track_key', 'track_loudness',
           'track_mode', 'track_speechiness', 'track_acousticness',
           'track_instrumentalness', 'track_liveness', 'track_valence',
           'track_tempo', 'track_duration_ms', 'track_time_signature',
           'track_name', 'track_date', 'track_popularity', 'track_preview',
           'track_correctness', 'artist_followers', 'artist_genres', 'artist_name',
           'artist_popularity', 'target', 'target_string', 'correcteness_value',
           'track_year', 'track_age'],
          dtype='object')




```python
X = df[['track_danceability', 'track_energy', 'track_key', 'track_loudness',
       'track_mode', 'track_speechiness', 'track_acousticness',
       'track_instrumentalness', 'track_liveness', 'track_valence',
       'track_tempo', 'track_duration_ms', 'track_time_signature',
       'track_name', 'track_popularity','artist_followers', 'artist_name',
       'artist_popularity', 'correcteness_value','track_age']]
```


```python
Y = df[['target']]
```


```python
X.to_pickle("data/X.pkl")
Y.to_pickle("data/Y.pkl")
```
