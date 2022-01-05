import numpy as np
import h5py
from pandas import concat, read_csv

fg_misses = [0,0,0,0,0,0,0,0,0,0]
fg_success = [0,0,0,0,0,0,0,0,0,0]
plays = read_csv('./nfl-big-data-bowl-2022/plays.csv')
print('load tracking')
tracking = concat([read_csv('./nfl-big-data-bowl-2022/tracking2018.csv'), read_csv('./nfl-big-data-bowl-2022/tracking2019.csv'), read_csv('./nfl-big-data-bowl-2022/tracking2020.csv')])
field_goals = []
for idx, row in plays.iterrows():
    if idx%100 == 0:
        print(str(idx)+'/'+str(len(plays)))
    if row['specialTeamsPlayType'] == 'Field Goal':
        if row['specialTeamsResult'] == 'Blocked Kick Attempt':
            row['specialTeamsResult'] = 'Kick Attempt No Good'
        #get kick length
        play = tracking[(tracking['gameId'] == row['gameId']) & (tracking['playId'] == row['playId'])]
        requiredYardage = row['absoluteYardlineNumber']
        
        if len(play) > 0 and (row['specialTeamsResult'] == 'Kick Attempt Good' or row['specialTeamsResult'] == 'Kick Attempt No Good'):
            if play.iloc[0]['playDirection'] == 'right':
                requiredYardage = 120 - row['absoluteYardlineNumber']
                
            field_goals.append([[requiredYardage + 7], row['specialTeamsResult']])

for f in field_goals:
    if f[1] == 'Kick Attempt Good':
        if f[0][0] < 20:
            fg_success[0] = fg_success[0] + 1
        elif f[0][0] < 25:
            fg_success[1] = fg_success[1] + 1
        elif f[0][0] < 30:
            fg_success[2] = fg_success[2] + 1
        elif f[0][0] < 35:
            fg_success[3] = fg_success[3] + 1
        elif f[0][0] < 40:
            fg_success[4] = fg_success[4] + 1
        elif f[0][0] < 45:
            fg_success[5] = fg_success[5] + 1
        elif f[0][0] < 50:
            fg_success[6] = fg_success[6] + 1
        elif f[0][0] < 55:
            fg_success[7] = fg_success[7] + 1
        elif f[0][0] < 60:
            fg_success[8] = fg_success[8] + 1
        else:
            fg_success[9] = fg_success[9] + 1    
    else:
        print(f[1])
        if f[0][0] < 20:
            fg_misses[0] = fg_misses[0] + 1
        elif f[0][0] < 25:
            fg_misses[1] = fg_misses[1] + 1
        elif f[0][0] < 30:
            fg_misses[2] = fg_misses[2] + 1
        elif f[0][0] < 35:
            fg_misses[3] = fg_misses[3] + 1
        elif f[0][0] < 40:
            fg_misses[4] = fg_misses[4] + 1
        elif f[0][0] < 45:
            fg_misses[5] = fg_misses[5] + 1
        elif f[0][0] < 50:
            fg_misses[6] = fg_misses[6] + 1
        elif f[0][0] < 55:
            fg_misses[7] = fg_misses[7] + 1
        elif f[0][0] < 60:
            fg_misses[8] = fg_misses[8] + 1
        else:
            fg_misses[9] = fg_misses[9] + 1 
yds = ['<20','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60+']
cum_makes = 0
cum_attempts = 0

print(fg_misses)
print(fg_success)
for i in range(len(fg_success)):
    cum_attempts = cum_attempts + fg_success[i] + fg_misses[i]
    cum_makes = cum_makes + fg_success[i]
    print(yds[i] + ' ' + str(fg_success[i]/(fg_success[i] + fg_misses[i])))
    print('Cumulative field goal percentage: ' + str(cum_makes/cum_attempts))