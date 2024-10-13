# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:16:02 2024

@author: devam
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:45:29 2024

@author: devam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import nba_api
from nba_api.stats.endpoints import draftcombinedrillresults
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import playerprofilev2
%matplotlib inline
#%%
years = []
for i in range(2001, 2018):
    years.append(i)
#%%
df_combine = pd.DataFrame(draftcombinedrillresults.DraftCombineDrillResults('00', 2000).get_data_frames()[0])
#%%
#Creating one large dataset 
for year in years:
    df_combine = pd.concat([df_combine, draftcombinedrillresults.DraftCombineDrillResults('00', year).get_data_frames()[0]], axis=0)

#%%
#Dropping empty row
df_combine = df_combine.drop('MODIFIED_LANE_AGILITY_TIME', axis=1)
#%%
# Preparing data
df_combine.reset_index(inplace=True)
df_combine.drop('index', axis=1, inplace=True)
df_combine.drop('TEMP_PLAYER_ID', axis=1, inplace=True)
#%%
# Deleting Empty Rows
index = []
for x in range(0, len(df_combine)):
    index.append(x)

for p in index:
    if (
        pd.isna(df_combine.loc[p, 'STANDING_VERTICAL_LEAP']) and
        pd.isna(df_combine.loc[p, 'MAX_VERTICAL_LEAP']) and
        pd.isna(df_combine.loc[p, 'LANE_AGILITY_TIME']) and
        pd.isna(df_combine.loc[p, 'THREE_QUARTER_SPRINT']) and
        pd.isna(df_combine.loc[p, 'BENCH_PRESS'])
    ):
        df_combine = df_combine.drop(p, axis=0)
#%%
#Resetting index again
df_combine = df_combine.reset_index()
df_combine = df_combine.drop('index', axis=1)
#%%
#Creating dataset with mean of all stats based on position
df_mean = df_combine.groupby("POSITION")[['STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS']].mean()

#%%
# Filling in missing values with mean of position
for p in range(0, 1122):
    if pd.isna(df_combine.loc[p, 'STANDING_VERTICAL_LEAP']):
        df_combine.loc[p, 'STANDING_VERTICAL_LEAP'] = df_mean.loc[df_combine.loc[p, 'POSITION'], 'STANDING_VERTICAL_LEAP']
    if pd.isna(df_combine.loc[p, 'MAX_VERTICAL_LEAP']):
        df_combine.loc[p, 'MAX_VERTICAL_LEAP'] = df_mean.loc[df_combine.loc[p, 'POSITION'], 'MAX_VERTICAL_LEAP']
    if pd.isna(df_combine.loc[p, 'LANE_AGILITY_TIME']):
        df_combine.loc[p, 'LANE_AGILITY_TIME'] = df_mean.loc[df_combine.loc[p, 'POSITION'], 'LANE_AGILITY_TIME']
    if pd.isna(df_combine.loc[p, 'BENCH_PRESS']):
        df_combine.loc[p, 'BENCH_PRESS'] = df_mean.loc[df_combine.loc[p, 'POSITION'], 'BENCH_PRESS']
    if pd.isna(df_combine.loc[p, 'THREE_QUARTER_SPRINT']):
        df_combine.loc[p, 'THREE_QUARTER_SPRINT'] = df_mean.loc[df_combine.loc[p, 'POSITION'], 'THREE_QUARTER_SPRINT']
#%%
df_comm = pd.DataFrame()  # Initialize an empty DataFrame
player_id = []
for player in df_combine['PLAYER_ID']:
    player_id.append(player)
#%%
#Creating dataframe with averages of career totals of players in combine history
pie = pd.DataFrame()
identity = []
for a in player_id:
    try:
        pie = pd.concat([pie, playerprofilev2.PlayerProfileV2(str(a)).get_data_frames()[0].groupby('LEAGUE_ID')[['PLAYER_ID', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].mean()])        
    except Exception as e:
        pass
#%%
pie.reset_index(inplace=True)
pie.drop('LEAGUE_ID', axis=1, inplace=True)
#%%
df_combine.set_index('PLAYER_ID', inplace=True)
pie.set_index('PLAYER_ID', inplace=True)
#%%
pie[['STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS']] = np.nan
#%%
#Combining combine stats and career stats
for p in player_id:
    if p in pie.index:
        pie.loc[p, ['STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS']] = df_combine.loc[p, ['STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS']]
        
#%%
#Making per 36 stats
pie[['FGM', 'FGA', 'FG3M', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']] = pie[['FGM', 'FGA', 'FG3M', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].div(pie['MIN'], axis=0)
pie[['FGM', 'FGA', 'FG3M', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']] = pie[['FGM', 'FGA', 'FG3M', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']]*48
#%%
#Renaming columns
pie.rename(columns={'FGM' : 'FGM_PER_48', 'FGA' : 'FGA_PER_48', 'FG3M' : 'FG3M_PER_48', 'FTM' : 'FTM_PER_48', 'FTA' : 'FTA_PER_48', 'OREB' : 'OREB_PER_48', 'DREB' : 'DREB_PER_48', 'REB' : 'REB_PER_48', 'AST' : 'AST_PER_48', 'STL' : 'STL_PER_48', 'BLK' : 'BLK_PER_48', 'TOV' : 'TOV_PER_48', 'PF' : 'PF_PER_48', 'PTS' : 'PTS_PER_48'}, inplace=True)
#%%
#Creating plot for lane agility and rebound
sns.regplot(data=pie, x='LANE_AGILITY_TIME', y='REB_PER_48')
plt.xlabel('Lane Agility Time (Sec)')
plt.ylabel('Rebounds Per 48 Min')
plt.title('Effect of Lane Agility Time on Rebounds')
plt.show()
#%%
#Creating plot for Lane_Agility_time and blocks
sns.regplot(data=pie, x='LANE_AGILITY_TIME', y='BLK_PER_48')
plt.xlabel('Lane Agility Time (Sec)')
plt.ylabel('Blocks Per 48 Min')
plt.title('Effect of Lane Agility Time on Blocks')
plt.show()
#%%
#Creating plot for Three Quarter Sprint and Reboudns
sns.regplot(data=pie, x='THREE_QUARTER_SPRINT', y='REB_PER_48')
plt.xlabel('THREE_QUARTER_SPRINT (Sec)')
plt.ylabel('Rebounds Per 48 Min')
plt.title('Effect of Three Quarter Sprint on Rebounds')
plt.show()

#%%
pie['SEASON_EXP'] = np.nan
#%%
#Adding how long each player was in the nba to the dataset
for p in player_id:
    if p in pie.index:
        try: 
            pie.loc[p, 'SEASON_EXP'] = commonplayerinfo.CommonPlayerInfo(str(p)).get_data_frames()[0].loc[0, 'SEASON_EXP']
        except Exception as e:
            print(f'error was {e}')
#%%
#Adding each player's position to compare by position
pie['POSITION'] = np.nan
for p in player_id:
    if p in pie.index:
        pie.loc[p, 'POSITION'] = df_combine.loc[p, 'POSITION']
#%%
#Creating a dataframe for guards
guard = pd.DataFrame(columns=pie.columns)

#%%
#Adding guards to "guard" dataframe
n = 0
for player in pie.index:
    if pie.loc[player, 'POSITION'] == 'PG' or pie.loc[player, 'POSITION'] == 'SG' or pie.loc[player, 'POSITION'] == 'SG-PG' or pie.loc[player, 'POSITION'] == 'PG-SG':
        guard.loc[n, ['MIN', 'FGM_PER_48', 'FGA_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT',
               'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48',
               'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48',
               'PF_PER_48', 'PTS_PER_48', 'STANDING_VERTICAL_LEAP',
               'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
               'BENCH_PRESS', 'SEASON_EXP', 'POSITION']] = pie.loc[player, ['MIN', 'FGM_PER_48', 'FGA_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT',
                      'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48',
                      'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48',
                      'PF_PER_48', 'PTS_PER_48', 'STANDING_VERTICAL_LEAP',
                      'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
                      'BENCH_PRESS', 'SEASON_EXP', 'POSITION']]
        n +=1
#%%
#Adding forwards to "forward" dataframe
forward = pd.DataFrame(columns=pie.columns)
n=0
for player in pie.index:
    if pie.loc[player, 'POSITION'] == 'SF' or pie.loc[player, 'POSITION'] == 'PF' or pie.loc[player, 'POSITION'] == 'SG-SF' or pie.loc[player, 'POSITION'] == 'SF-SG' or pie.loc[player, 'POSITION'] == 'SF-PF' or pie.loc[player, 'POSITION'] == 'PF-SF':
        forward.loc[n, ['MIN', 'FGM_PER_48', 'FGA_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT',
               'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48',
               'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48',
               'PF_PER_48', 'PTS_PER_48', 'STANDING_VERTICAL_LEAP',
               'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
               'BENCH_PRESS', 'SEASON_EXP', 'POSITION']] = pie.loc[player, ['MIN', 'FGM_PER_48', 'FGA_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT',
                      'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48',
                      'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48',
                      'PF_PER_48', 'PTS_PER_48', 'STANDING_VERTICAL_LEAP',
                      'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
                      'BENCH_PRESS', 'SEASON_EXP', 'POSITION']]
                                                                            
        n += 1
#%%
#Adding centers to "center" dataframe
center = pd.DataFrame(columns=pie.columns)
n=0
for player in pie.index:
    if pie.loc[player, 'POSITION'] == 'C' or pie.loc[player, 'POSITION'] == 'PF-C' or pie.loc[player, 'POSITION'] == 'C-PF':
        center.loc[n, ['MIN', 'FGM_PER_48', 'FGA_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT',
               'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48',
               'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48',
               'PF_PER_48', 'PTS_PER_48', 'STANDING_VERTICAL_LEAP',
               'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
               'BENCH_PRESS', 'SEASON_EXP', 'POSITION']] = pie.loc[player, ['MIN', 'FGM_PER_48', 'FGA_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT',
                      'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48',
                      'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48',
                      'PF_PER_48', 'PTS_PER_48', 'STANDING_VERTICAL_LEAP',
                      'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT',
                      'BENCH_PRESS', 'SEASON_EXP', 'POSITION']]
                                                                            
        n += 1

#%%
#Getting correlation values within postions
guard_corr = pd.DataFrame(guard.drop('POSITION', axis=1)).corr()
forward_corr = pd.DataFrame(forward.drop('POSITION', axis=1)).corr()
center_corr = pd.DataFrame(center.drop('POSITION', axis=1)).corr()
#%%
#Changing datatypes so regression plot can be made
guard[['MIN', 'PTS_PER_48', 'FGA_PER_48', 'FGM_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT', 'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48', 'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48', 'PF_PER_48', 'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS', 'SEASON_EXP']] = guard[['MIN', 'PTS_PER_48', 'FGA_PER_48', 'FGM_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT', 'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48', 'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48', 'PF_PER_48', 'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS', 'SEASON_EXP']].astype('float')
forward[['MIN', 'PTS_PER_48', 'FGA_PER_48', 'FGM_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT', 'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48', 'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48', 'PF_PER_48', 'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS', 'SEASON_EXP']] = forward[['MIN', 'PTS_PER_48', 'FGA_PER_48', 'FGM_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT', 'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48', 'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48', 'PF_PER_48', 'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS', 'SEASON_EXP']].astype('float')
center[['MIN', 'PTS_PER_48', 'FGA_PER_48', 'FGM_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT', 'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48', 'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48', 'PF_PER_48', 'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS', 'SEASON_EXP']] = center[['MIN', 'PTS_PER_48', 'FGA_PER_48', 'FGM_PER_48', 'FG_PCT', 'FG3M_PER_48', 'FG3_PCT', 'FTM_PER_48', 'FTA_PER_48', 'FT_PCT', 'OREB_PER_48', 'DREB_PER_48', 'REB_PER_48', 'AST_PER_48', 'STL_PER_48', 'BLK_PER_48', 'TOV_PER_48', 'PF_PER_48', 'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS', 'SEASON_EXP']].astype('float')

#%%
#Creating regression plot for guard
sns.regplot(data=guard, x='THREE_QUARTER_SPRINT', y='SEASON_EXP')
plt.xlabel('Three Quarter Sprint (sec)')
plt.ylabel('Seasons Played')
plt.title('Effect of Speed on Seasons Played')
plt.show()
#%%
#making linear regression model for guards
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
regr = linear_model.LinearRegression()

X = np.asanyarray(guard[['STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS']])
y = np.asanyarray(guard['SEASON_EXP'])
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
#%%
#Making a linear regression model for forwards
X = np.asanyarray(forward[['STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS']])
y = np.asanyarray(forward['SEASON_EXP'])
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
print(f'The coefficients are: {regr.coef_}')
#%%
#Making a linear regression model for centers
X = np.asanyarray(center[['STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT', 'BENCH_PRESS']])
y = np.asanyarray(center['SEASON_EXP'])
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
print(f'The coefficients are: {regr.coef_}')
