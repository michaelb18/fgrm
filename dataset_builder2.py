import math
from os import read
import numpy as np
from pandas import concat, read_csv
import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException
import time
import json
from datetime import datetime
import random
#orientation of each stadium relative to north
stadium_orienter = {
    
}
# add 360-azi to wind dir to get wind azi
def fill_stadium_azi(stadium_coordinates):
    for idx, stadium in stadium_coordinates.iterrows():
        stadium_orienter[stadium.StadiumName] = (stadium.RoofType, stadium.StadiumAzimuthAngle)

def get_wind_vector(gameId, time, games, weather):
    game = weather[weather['GameId'] == gameId]
    stadium = games[games['GameId'] == gameId]['StadiumName']
    roof_type, orientation = stadium_orienter[stadium.values[0]]
    c = 0
    if len(game) == 0:
        return None, None, None
    while c < len(game)-1 and game.TimeMeasure.values[c] < time:
        c = c + 1
    game = game.iloc[c]
    if (game.Precipitation > 0 or game.Temperature < 40 or game.WindSpeed > 40)and roof_type == 'Retractable':
        roof_type = 'Indoor'
    if roof_type == 'Indoor':
        game.Precipitation = 0
        game.WindSpeed = 0
    return game.WindSpeed, game.WindDirection - orientation, game.Precipitation
game_weather = read_csv('./archive/games_weather.csv')
stadiums = read_csv('./archive/games.csv')
stadium_coordinates = read_csv('./archive/stadium_coordinates.csv')
game_weather.dropna(inplace=True, how='any')
stadiums.dropna(inplace=True, how='any')
stadium_coordinates.dropna(inplace=True, how='any')
fill_stadium_azi(stadium_coordinates)
games = read_csv('./nfl-big-data-bowl-2022/games.csv')
plays = read_csv('./nfl-big-data-bowl-2022/plays.csv')
print('load tracking')
tracking = concat([read_csv('./nfl-big-data-bowl-2022/tracking2018.csv'), read_csv('./nfl-big-data-bowl-2022/tracking2019.csv'), read_csv('./nfl-big-data-bowl-2022/tracking2020.csv')])
kickers = {}
field_goals = []
weather_conditions = {}
print('Starting')
print(plays.columns)
print(tracking.columns)
misses_data = []
n_misses = 0
n_hits = 0
for idx, row in plays.iterrows():
    if idx%100 == 0:
        print(str(idx)+'/'+str(len(plays)))
    if row['specialTeamsPlayType'] == 'Field Goal':
        if row['specialTeamsResult'] == 'Blocked Kick Attempt':
            row['specialTeamsResult'] = 'Kick Attempt No Good'
        #get kick length
        play = tracking[(tracking['gameId'] == row['gameId']) & (tracking['playId'] == row['playId'])]
        requiredYardage = row['absoluteYardlineNumber']
        
        team = games[games['gameId'] == row['gameId']]['homeTeamAbbr'].values[0]
        if len(play) > 0 and (row['specialTeamsResult'] == 'Kick Attempt Good' or row['specialTeamsResult'] == 'Kick Attempt No Good'):
            wind_speed, deg, precip = get_wind_vector(row['gameId'], play.iloc[0].time, stadiums, game_weather)
            if wind_speed is not None:
                wind_sign = 1
                if play.iloc[0]['playDirection'] == 'right':
                    requiredYardage = 120 - row['absoluteYardlineNumber']
                    wind_sign = -1
                if row['kickerId'] not in kickers:
                    kickers[row['kickerId']] = [0, 0]
                kickers[row['kickerId']][1] = kickers[row['kickerId']][1] + 1
                if row['specialTeamsResult'] == 'Kick Attempt Good':
                    kickers[row['kickerId']][0] = kickers[row['kickerId']][0] + 1
                    n_hits = n_hits + 1
                else:
                    misses_data.append([requiredYardage + 7, wind_sign * wind_speed * math.sin(deg * math.pi/180), wind_sign * wind_speed * math.cos(deg * math.pi/180), kickers[row['kickerId']][0]/kickers[row['kickerId']][1]*100, precip])
                    n_misses = n_misses + 1
                field_goals.append([[requiredYardage + 7, wind_sign * wind_speed * math.sin(deg * math.pi/180), wind_sign * wind_speed * math.cos(deg * math.pi/180), kickers[row['kickerId']][0]/kickers[row['kickerId']][1]*100, precip], row['specialTeamsResult']])

def euclidean_distance(a, b):
    c = np.array(a)
    d = np.array(b)
    f = c - d
    return np.linalg.norm(c-d)

def get_knn(x, distance_metric, D, k = 1):
    nearest_indices = []
    for i in range(k):
        nearest_indices.append([-1, math.inf])
    nearest_indices = np.array(nearest_indices)
    for v in range(len(D)):
        nearest = False
        for n in nearest_indices:
            if distance_metric(D[v], x) < n[1] and distance_metric(D[v], x) != 0:
                nearest = True
                break
        if nearest:
            nearest_indices[k-1] = [v, distance_metric(D[v], x)]
            nearest_indices = nearest_indices[np.argsort(nearest_indices[:, 1])]
            
    return nearest_indices


for _ in range(n_hits - n_misses):
    e = np.array(random.choice(misses_data))
    i = get_knn(e, euclidean_distance, misses_data)
    nn = np.array(misses_data[int(i[0][0])])
    new_element = e + (nn - e) * random.uniform(0, 1)
    field_goals.append([new_element, 'Kick Attempt No Good'])
field_goals = np.stack(field_goals)
with open('saved_fg.npy', 'wb') as f:
    np.save(f, field_goals)
print(len(field_goals))