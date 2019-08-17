---
layout: post
title:  Spotify Music Analysis - Part 1
tags: API spotify spotipy python data-mining
---


Is a machine-learning model able to predict my musical tastes? That is the question behind this project. Music lover in my heart, music has been the rhythm of my life since my earliest years. From rock to rap, classical to folk, I have a strong taste for eclecticism. What better way to answer to that question than by exploring the Spotify API in more depth?


#### Objective of the notebook
Our goal is to show how to collect audio features data for tracks from the official Spotify Web API for further analysis. As mentioned above, each step of the process will have its own notebook associated with it

## 1 - Setting Up


```python
#Import of packages
import pandas as pd 
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

#Credentials
cid ="**************************************" 
secret = "**********************************" 

#Authentification
sp = spotipy.Spotify() 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
sp.trace=False 
```

## 2 - Get data from songs playlists

Separate playlists containing songs that I like and others that I don't like (as diversified as possible) have been created upstream to recover enough data taking into account the limitations of the API (the limit of songs recovered in each playlist is 100). From there, we need to retrieve as much information as possible, namely data related to the song, the artist or the album, so that we can identify the attributes that best characterize my musical tastes.


![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/spotify/spotify_crawler_1.PNG)

<p>
  
  </p>

Thanks to the web API endpoints, any user can access to the Spotify catalog and user data. 
[Here is a list of the different types of data that can be recovered.](https://developer.spotify.com/documentation/web-api/reference/) Given the purpose of the project, namely modelling, the objective here was to recover as much information as possible, even if it meant that some of it would be excluded later on. In the end, only 4 endpoints (i.e. Albums, Tracks, Artists and Playlists) out of the 50 available were used. We will discuss our approach to variable selection later on.

<p>
  
  </p>

![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/spotify/spotify_crawler_2.PNG)

<p>
  
  </p>

![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/spotify/spotify_crawler_3.PNG)

<p>
  
  </p>


```python
hated_songs = get_data_from_list_of_playlists(bad_playlist_ids)
```

    For playlist 1 / 26 : we have  100  songs
    Number of tracks remaining before the break :  100
    * * * ----------------------------------------------------- * * *
    For playlist 2 / 26 : we have  50  songs
    Number of tracks remaining before the break :  150
    * * * ----------------------------------------------------- * * *
    For playlist 3 / 26 : we have  40  songs
    Number of tracks remaining before the break :  190
    * * * ----------------------------------------------------- * * *
    For playlist 4 / 26 : we have  40  songs
    Number of tracks remaining before the break :  230
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 5 / 26 : we have  61  songs
    Number of tracks remaining before the break :  61
    * * * ----------------------------------------------------- * * *
    For playlist 6 / 26 : we have  61  songs
    Number of tracks remaining before the break :  122
    * * * ----------------------------------------------------- * * *
    For playlist 7 / 26 : we have  50  songs
    Number of tracks remaining before the break :  172
    * * * ----------------------------------------------------- * * *
    For playlist 8 / 26 : we have  50  songs
    Number of tracks remaining before the break :  222
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 9 / 26 : we have  39  songs
    Number of tracks remaining before the break :  39
    * * * ----------------------------------------------------- * * *
    For playlist 10 / 26 : we have  50  songs
    Number of tracks remaining before the break :  89
    * * * ----------------------------------------------------- * * *
    For playlist 11 / 26 : we have  60  songs
    Number of tracks remaining before the break :  149
    * * * ----------------------------------------------------- * * *
    For playlist 12 / 26 : we have  50  songs
    Number of tracks remaining before the break :  199
    * * * ----------------------------------------------------- * * *
    For playlist 13 / 26 : we have  100  songs
    Number of tracks remaining before the break :  299
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 14 / 26 : we have  60  songs
    Number of tracks remaining before the break :  60
    * * * ----------------------------------------------------- * * *
    For playlist 15 / 26 : we have  100  songs
    Number of tracks remaining before the break :  160
    * * * ----------------------------------------------------- * * *
    For playlist 16 / 26 : we have  50  songs
    Number of tracks remaining before the break :  210
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 17 / 26 : we have  100  songs
    Number of tracks remaining before the break :  100
    * * * ----------------------------------------------------- * * *
    For playlist 18 / 26 : we have  50  songs
    Number of tracks remaining before the break :  150
    * * * ----------------------------------------------------- * * *
    For playlist 19 / 26 : we have  100  songs
    Number of tracks remaining before the break :  250
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 20 / 26 : we have  50  songs
    Number of tracks remaining before the break :  50
    * * * ----------------------------------------------------- * * *
    For playlist 21 / 26 : we have  47  songs
    Number of tracks remaining before the break :  97
    * * * ----------------------------------------------------- * * *
    For playlist 22 / 26 : we have  80  songs
    Number of tracks remaining before the break :  177
    * * * ----------------------------------------------------- * * *
    For playlist 23 / 26 : we have  70  songs
    Number of tracks remaining before the break :  247
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 24 / 26 : we have  52  songs
    Number of tracks remaining before the break :  52
    * * * ----------------------------------------------------- * * *
    For playlist 25 / 26 : we have  100  songs
    Number of tracks remaining before the break :  152
    * * * ----------------------------------------------------- * * *
    For playlist 26 / 26 : we have  100  songs
    Number of tracks remaining before the break :  252
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    


```python
loved_songs = get_data_from_list_of_playlists(good_playlist_ids)
```

    For playlist 1 / 26 : we have  31  songs
    Number of tracks remaining before the break :  31
    * * * ----------------------------------------------------- * * *
    For playlist 2 / 26 : we have  100  songs
    Number of tracks remaining before the break :  131
    * * * ----------------------------------------------------- * * *
    For playlist 3 / 26 : we have  33  songs
    Number of tracks remaining before the break :  164
    * * * ----------------------------------------------------- * * *
    For playlist 4 / 26 : we have  100  songs
    Number of tracks remaining before the break :  264
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 5 / 26 : we have  24  songs
    Number of tracks remaining before the break :  24
    * * * ----------------------------------------------------- * * *
    For playlist 6 / 26 : we have  90  songs
    Number of tracks remaining before the break :  114
    * * * ----------------------------------------------------- * * *
    For playlist 7 / 26 : we have  100  songs
    Number of tracks remaining before the break :  214
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 8 / 26 : we have  61  songs
    Number of tracks remaining before the break :  61
    * * * ----------------------------------------------------- * * *
    For playlist 9 / 26 : we have  100  songs
    Number of tracks remaining before the break :  161
    * * * ----------------------------------------------------- * * *
    For playlist 10 / 26 : we have  51  songs
    Number of tracks remaining before the break :  212
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 11 / 26 : we have  58  songs
    Number of tracks remaining before the break :  58
    * * * ----------------------------------------------------- * * *
    For playlist 12 / 26 : we have  50  songs
    Number of tracks remaining before the break :  108
    * * * ----------------------------------------------------- * * *
    For playlist 13 / 26 : we have  50  songs
    Number of tracks remaining before the break :  158
    * * * ----------------------------------------------------- * * *
    For playlist 14 / 26 : we have  100  songs
    Number of tracks remaining before the break :  258
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 15 / 26 : we have  100  songs
    Number of tracks remaining before the break :  100
    * * * ----------------------------------------------------- * * *
    For playlist 16 / 26 : we have  100  songs
    Number of tracks remaining before the break :  200
    * * * ----------------------------------------------------- * * *
    For playlist 17 / 26 : we have  29  songs
    Number of tracks remaining before the break :  229
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 18 / 26 : we have  57  songs
    Number of tracks remaining before the break :  57
    * * * ----------------------------------------------------- * * *
    For playlist 19 / 26 : we have  100  songs
    Number of tracks remaining before the break :  157
    * * * ----------------------------------------------------- * * *
    For playlist 20 / 26 : we have  100  songs
    Number of tracks remaining before the break :  257
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 21 / 26 : we have  89  songs
    Number of tracks remaining before the break :  89
    * * * ----------------------------------------------------- * * *
    For playlist 22 / 26 : we have  80  songs
    Number of tracks remaining before the break :  169
    * * * ----------------------------------------------------- * * *
    For playlist 23 / 26 : we have  67  songs
    Number of tracks remaining before the break :  236
    It's time to take a little nap
    * * * ----------------------------------------------------- * * *
    For playlist 24 / 26 : we have  50  songs
    Number of tracks remaining before the break :  50
    * * * ----------------------------------------------------- * * *
    For playlist 25 / 26 : we have  100  songs
    Number of tracks remaining before the break :  150
    * * * ----------------------------------------------------- * * *
    For playlist 26 / 26 : we have  33  songs
    Number of tracks remaining before the break :  183
    * * * ----------------------------------------------------- * * *
    


```python
#Drop duplicates for all the dataframes
hated_dataframe = hated_songs.drop_duplicates(keep = 'first', subset='track_id') 
loved_dataframe = loved_songs.drop_duplicates(keep = 'first', subset='track_id') 
```


```python
#Print shapes of dataframes
print(hated_dataframe.shape)
print(loved_dataframe.shape)
```

    (1625, 33)
    (1811, 33)
    

## 3 - Sampling 500 rows for each dataframe


```python
sampled_hated_dataframe = hated_dataframe.sample(n=500)
sampled_loved_dataframe = loved_dataframe.sample(n=500)
```


```python
sampled_hated_dataframe.to_pickle("data/hated_dataframe.pkl")
sampled_loved_dataframe.to_pickle("data/loved_dataframe.pkl")
```
