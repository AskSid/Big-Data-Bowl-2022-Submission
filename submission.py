#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from pandas import Series
pd.set_option('max_columns', None)
pd.set_option('display.max_colwidth', None)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from tensorflow.keras.models import Model

from matplotlib import pyplot as plt 

import plotly.express as px
import plotly.graph_objects as go


# In[2]:


# importing the data

tracking_2018 = pd.read_csv('BigDataBowl2022/tracking2018.csv')
tracking_2019 = pd.read_csv('BigDataBowl2022/tracking2019.csv')
tracking_2020 = pd.read_csv('BigDataBowl2022/tracking2020.csv')
years = [tracking_2018, tracking_2019, tracking_2020]
tracking = pd.concat(years)
plays    = pd.read_csv('BigDataBowl2022/plays.csv')
players = pd.read_csv('BigDataBowl2022/players.csv')


# In[3]:


# getting all game/play ids that are punts from pff data

plays_punts = plays[(plays.specialTeamsPlayType == 'Punt') & (plays.specialTeamsResult == 'Return')]
plays_punts['id'] = plays_punts['gameId'].astype(str) + plays_punts['playId'].astype(str)
plays_punts = plays_punts[plays_punts['kickReturnYardage'].notna()]
punt_ids = plays_punts['id'].to_list()

# keeping plays in tracking data that are punts

tracking['id'] = tracking['gameId'].astype(str) + tracking['playId'].astype(str)
tracking = tracking[tracking['id'].isin(punt_ids)]
print(plays_punts.shape)

# drop football tracking data
tracking = tracking[tracking.displayName != 'football']
tracking['nflId'] = pd.to_numeric(tracking['nflId'], downcast='integer')


# In[4]:


# get actual yards remaining for each play

id_group = tracking.groupby('id')
id_group_test = tracking[tracking.id == '2021010315175'].groupby('id')

yardages_train = []
distance_matrices_train = []
returner_params_train = []

test_frame_info = []
yardages_test = []
distance_matrices_test = []
returner_params_test = []
id_test = []

def add_yardage(group):
    try:        
        # keep relevant frames from a play
        start_frame_id = group[(group.event == 'punt_received')].iloc[0]['frameId']
        end_frame_id = group[(group.event == 'out_of_bounds') | (group.event == 'tackle') | (group.event == 'fumble') | (group.event == 'touchdown')].iloc[0]['frameId']
        group = group[(group['frameId'] >= start_frame_id) & (group['frameId'] <= end_frame_id)]

        # get returner's starting x value (beginning of punt return)
        returnerId = plays_punts[plays_punts.id == group.iloc[0]['id']].iloc[0]['returnerId']
        returner_data = group[group.nflId == int(returnerId)].sort_values(['frameId'])
        returner_start_x = returner_data[returner_data.frameId == start_frame_id].iloc[0]['x']
        
        
        total_yardage = plays_punts[plays_punts.id == group.iloc[0]['id']].iloc[0]['kickReturnYardage']
        
        # boolean value for returner
        group['isReturner'] = (group['nflId'] == int(returnerId)).astype(int)
        returner_team = returner_data.iloc[0]['team']
        group['returnTeam'] = (group['team'] == returner_team).astype(int)
        
        # standardize speed and acceleration
        group['s'] = group['s'] / 20
        group['a'] = group['a'] / 10
        
        # get returner's yardage remaining in the play and adjust orientation and direction based on play direction
        if group.iloc[0]['playDirection'] == 'right':
            play_direction = 'right'
            returner_end_x = group[group.nflId == int(returnerId)]['x'].min()
            group['yardageLeft'] = total_yardage - (returner_start_x - group['x'])
            
            group['o'] = group['o'] / 360
            group['dir'] = group['dir'] / 360
        else:
            returner_end_x = group[group.nflId == int(returnerId)]['x'].max()
            group['yardageLeft'] = total_yardage - (group['x'] - returner_start_x)
            play_direction = 'left'
            
            group.loc[group['o'] <= 180, 'o'] = (group['o'] + 180) / 360
            group.loc[group['o'] >= 180, 'o'] = (group['o'] - 180) / 360
            
            group.loc[group['dir'] <= 180, 'dir'] = (group['dir'] + 180) / 360
            group.loc[group['dir'] >= 180, 'dir'] = (group['dir'] - 180) / 360
        
        
        
        # group each frame of a play and set x and y to be relative to returner
        by_time = group.groupby('time')
        for name, group_by_time in by_time:
            returner_current_x = group_by_time[group_by_time.isReturner == True].iloc[0]['x']
            returner_current_y = group_by_time[group_by_time.isReturner == True].iloc[0]['y']
            
            curr_id = group_by_time.iloc[0]['id']
            curr_frame_id = group_by_time.iloc[0]['frameId']
            
            # final Y value
            yardage = group_by_time[group_by_time.isReturner == True].iloc[0]['yardageLeft']

            group_by_time = group_by_time[['x', 'y', 's', 'a', 'o', 'dir', 'isReturner', 'returnTeam']].sort_values(['isReturner', 'returnTeam', 'x'])            
            returner_frame_param = group_by_time[group_by_time.isReturner == True][['s', 'a', 'o', 'dir']]
            

            # distance matrix computation
            x_coords = group_by_time['x'].values
            y_coords = group_by_time['y'].values
            
            temp_distance_matrix = np.zeros((22, 22))
            for i in range(22):
                for j in range(22):
                    t = (x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2
                    temp_distance_matrix[i, j] = t
                    temp_distance_matrix[j, i] = t

            if (str(group.iloc[0]['id'])[0:4] == '2018') or (str(group.iloc[0]['id'])[0:4] == '2019'):
                distance_matrices_train.append(temp_distance_matrix)
                returner_params_train.append(returner_frame_param)
                yardages_train.append(yardage)
            else:
                distance_matrices_test.append(temp_distance_matrix)
                returner_params_test.append(returner_frame_param)
                yardages_test.append(yardage)
                test_frame_info.append([curr_id + str(curr_frame_id), play_direction])
                id_test.append(curr_id)

    # exclude plays where this doesn't work (most likely nans or empty/missing plays)
    except:
        try:
            print(group.iloc[0].id)
        except:
            print('empty play')

id_group.apply(add_yardage)


# In[5]:


# model 3: combined distance matrix and returner parameters

returner_params_train = np.array(returner_params_train).astype('float32')
returner_params_test = np.array(returner_params_test).astype('float32')
id_test = np.array(id_test)


distance_matrices_train = np.array(distance_matrices_train).astype('float32')
distance_matrices_test = np.array(distance_matrices_test).astype('float32')

yardages_train = np.array(yardages_train).astype('float32')
yardages_test = np.array(yardages_test).astype('float32')

returner_params_train = returner_params_train.reshape(64973, 4)
returner_params_test = returner_params_test.reshape(28088, 4)


X3_train, X3_test, X4_train, X4_test, Y2_train, Y2_test = train_test_split(distance_matrices_train, returner_params_train, yardages_train, test_size = 0.25)



# In[6]:


def create_returner_model(dim):
    model3 = Sequential()
    model3.add(Dense(8, input_dim = dim, activation = 'relu'))
    model3.add(Dense(4, activation = 'relu'))
    model3.add(Dense(1, activation = 'relu'))
    return model3

def create_distance_matrix_model():
    model4 = Sequential()
    model4.add(Conv2D(12, (3, 3), activation='relu', input_shape=(22, 22, 1)))
    model4.add(BatchNormalization(axis = -1))
    model4.add(MaxPooling2D((2, 2)))
    model4.add(Conv2D(4, (2, 2), activation='relu'))
    model4.add(Flatten())
    model4.add(Dense(22, activation = 'relu'))
    model4.add(Dense(22, activation = 'relu'))
    model4.add(Dense(22, activation = 'relu'))
    model4.add(Dense(22, activation = 'relu'))
    model4.add(Dense(22, activation = 'relu'))
    model4.add(Dense(4, activation = 'relu'))
    return model4

returner_model = create_returner_model(4)
distance_matrix_model = create_distance_matrix_model()
combined = concatenate([returner_model.output, distance_matrix_model.output])
x = Dense(4, activation = 'relu')(combined)
x = Dense(1, activation = 'linear')(x)
model5 = Model(inputs=[returner_model.input, distance_matrix_model.input], outputs=x)

model5.compile(optimizer='Adam', loss='mse')
model5.fit(x=[X4_train, X3_train], y=Y2_train, epochs = 45)


# In[7]:


test_acc2 = model5.evaluate(x = [X4_test, X3_test], y = Y2_test, verbose = 2)

print('\nTest accuracy:', test_acc2)


# In[57]:


fig = px.scatter(tracking[tracking.id == '2020091302137'], x="x", y="y", animation_frame="frameId", color="team")
fig.update_xaxes(range=[0,120])
fig.update_yaxes(range=[0, 53.3])
fig.show()


# In[64]:


# averaged inverse of the difference in expected yards between optimal direction and chosen direction

def orientation_yards_metric(index, show_plot):
    returner_info = returner_params_test[index:index+1]
    frame_prediction = []
    for i in range(360):
        val = (i + 1) / 360
        prediction = model5.predict(x = [np.array([returner_info[0][0], returner_info[0][1], returner_info[0][2], val]).reshape(1, 4), distance_matrices_test[index:index+1]])
        frame_prediction.append(prediction.item())
    x = np.arange(0, 360)
    frame_prediction = np.array(frame_prediction)
        
    returner_dir = returner_info[0][3] * 360
    if test_frame_info[index:index+1][0][1] == 'left':
        frame_prediction = np.concatenate((frame_prediction[179:360], frame_prediction[0:180]))
        if (returner_info[0][3] * 360) > 180:
            returner_dir = returner_dir - 180
        else:
            returner_dir = returner_dir + 180
        
    if show_plot:
        
        plt.plot(frame_prediction, linestyle = 'dotted')
        plt.axvline(x=returner_dir)
        max_y = max(frame_prediction)
        max_x = x[frame_prediction.argmax()]
        plt.axvline(x=max_x, color='red')
        
        print("actual decision: direction of " + str(round(returner_dir)) + " degrees")
        print("recommended decision: direction of " + str(max_x) + " degrees")
    else:
        return (abs(max(frame_prediction) - frame_prediction[round(returner_dir)])) / abs(max(frame_prediction))

print(orientation_yards_metric(2525, False))
plays_punts_2020_2021 = plays_punts[plays_punts.id.str[0:3] == '202']
id_test = np.array(id_test)



def returner_play_decision_metric(gId, pId):
    game = plays_punts_2020_2021[plays_punts_2020_2021.gameId == gId]
    play = game[game.returnerId == str(pId)]
    
    play['id'] = play['gameId'].astype(str) + play['playId'].astype(str)
    returner_play_ids = play['id'].to_numpy()

    frame_indices = np.array(np.where(np.in1d(id_test, returner_play_ids)))[0]
    print(frame_indices.shape)
    counter = 0
    sum = 0
    for index in frame_indices:
        counter = counter + 1
        metric = orientation_yards_metric(index, False)
        sum = sum + metric
    return sum / (counter + 1)

print(returner_play_decision_metric(2021010313, 52631))

