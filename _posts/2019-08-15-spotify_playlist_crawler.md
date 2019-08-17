---
layout: post
title:  Spotify Music Analysis - Part 1
tags: spotify spotipy python jupyter-notebook machine-learning 
---


Is a machine-learning model able to predict my musical tastes? That is the question behind this project. Music lover in my heart, music has been the rhythm of my life since my earliest years. From rock to rap, classical to folk, I have a strong taste for eclecticism. What better way to answer to that question than by exploring the Spotify API in more depth?


## Collecting data from the Spotify Web API using Spotipy

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


```python
bad_playlist_ids = ["37i9dQZF1DXaiEFNvQPZrM","37i9dQZF1DXb7WmotStdsj",
                    "37i9dQZF1DWTUfv2yzHEe7","37i9dQZF1DXcSPhLAnCjoM",
                    "37i9dQZF1DXdF699XuZIvg","37i9dQZF1DX14fiWYoe7Oh",
                    "37i9dQZF1DWVuV87wUBNwc","37i9dQZF1DX32nf7PAbnUl",
                    "2tSrDVaFN05VmQtWt4RS2U","37i9dQZF1DXd0Y4aXXQXWv",
                    "37i9dQZF1DWWl7MndYYxge","37i9dQZF1DWVAWlq3l00p0",
                    "37i9dQZF1DX0BcQWzuB7ZO","37i9dQZF1DX6J5NfMJS675",
                    "37i9dQZF1DXaXB8fQg7xif","37i9dQZF1DXc3KygMa1OE7",
                    "37i9dQZF1DX1X7WV84927n","37i9dQZF1DX7LGssahBoms",
                    "37i9dQZF1DX82pCGH5USnM","37i9dQZF1DX9GEHeZm41j4",
                    "37i9dQZF1DXc7KgLAqOCoC","37i9dQZF1DX5Q27plkaOQ3",
                    "37i9dQZF1DWWOaP4H0w5b0","37i9dQZF1DXe6bgV3TmZOL",
                    "1gcxl3Byd85WxwvIx2dBGO","6xnfRIFfuYOfWqJivjUPNz"]


good_playlist_ids = ["5MP9Pm7rfzpCV7VilNOqVO","0pzDTD34V3iEmMXshTWi75",
                     "1Wl63oltvnbXSMhQg2Afgb","4yuDi9FA9ApDkIILZYkPYL",
                     "5Q8Njo6JpRFMl340AAd5OD","5F9pa8Dr5uK9N8itsmLa1f",
                     "25lSdov27cHfJLkd9GwZZ0","37i9dQZF1DXawlg2SZawZf",
                     "37i9dQZF1DX186v583rmzp","37i9dQZF1DX7Mq3mO5SSDc",
                     "37i9dQZF1DX4AyFl3yqHeK","37i9dQZF1DXbITWG1ZJKYt",
                     "37i9dQZF1DX1S1NduGwpsa","0SE5dQTI6BRvtCbsxgKOmh",
                     "3uJ6lLXLypKkM61Rl8vJKu","3svGXFgifoYT9jEjkKindx",
                     "37i9dQZF1DWWrHouBoNlTS","37i9dQZF1DX6K3W8KBiALe",
                     "37i9dQZF1DXah8e1pvF5oE","37i9dQZF1DX9G9wwzwWL2k",
                     "37i9dQZF1DWUoqEG4WY6ce","37i9dQZF1DX2Nc3B70tvx0",
                     "37i9dQZF1DWWM6GBnxtToT","37i9dQZF1DWVsh2vXzlKFb",
                     "37i9dQZF1DX2sUQwD7tbmL","37i9dQZF1DWZq91oLsHZvy"]
```


```python
def get_data_from_one_playlist(playlistid):
    
    playlist = sp.user_playlist("msleonies", playlistid)

    songs = playlist["tracks"]["items"] 
    track_features = []
    for i in range(len(songs)):
        track_id = songs[i]["track"]["id"]
        track_name = songs[i]["track"]["name"]
        track_date = songs[i]["track"]["album"]["release_date"]
        track_popularity = songs[i]["track"]["popularity"]
        track_preview = songs[i]["track"]["preview_url"]
        track_correctness = songs[i]["track"]["explicit"]
        artist_id = songs[i]["track"]["artists"][0]["id"]
        allid = (track_id, track_name, track_date, 
                      track_popularity, track_preview, track_correctness, 
                      artist_id)
        track_features.append(allid)

    track_features_df = pd.DataFrame(sp.audio_features([item[0] for item in track_features]))
    track_features_df = track_features_df.add_prefix('track_')
    track_name_df = pd.Series([item[1] for item in track_features]).to_frame(name='track_name')
    track_date_df = pd.Series([item[2] for item in track_features]).to_frame(name='track_date')
    track_popularity_df = pd.Series([item[3] for item in track_features]).to_frame(name='track_popularity')
    track_preview_df = pd.Series([item[4] for item in track_features]).to_frame(name='track_preview')
    track_correctness_df = pd.Series([item[5] for item in track_features]).to_frame(name='track_correctness')
    
    artist_df = pd.DataFrame([sp.artist(item) for item in [row[-1] for row in track_features]])
    artist_df = artist_df.add_prefix('artist_')
    
    playlistdataframe = pd.concat((track_features_df,track_name_df,
                                   track_date_df,track_popularity_df,
                                   track_preview_df,
                                   track_correctness_df,
                                   artist_df),axis=1)

    return playlistdataframe
```


```python
#After a first try, we integrate our function created above and 
#then loop through a list of playlists 
def get_data_from_list_of_playlists(playlist_ids):
    songs_dataframe = []
    total_tracks = 0
    for idx,playlist_id in enumerate(playlist_ids):
        playlist_data = get_data_from_one_playlist(playlist_id)
        total_tracks += len(playlist_data)
        songs_dataframe.append(playlist_data)
        print("For playlist" + ' ' + str(idx+1),"/",str(len(playlist_ids)) + ' ' + ":" 
              + ' ' + "we have" + ' ', str(len(playlist_data)) + ' ',"songs")
        print("Number of tracks remaining before the break : ", total_tracks)
        if total_tracks > 200:
            print("It's time to take a little nap")
            time.sleep(30)
            total_tracks = 0
        print("* * * ----------------------------------------------------- * * *")
    return pd.concat(songs_dataframe,axis=0)
```


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
