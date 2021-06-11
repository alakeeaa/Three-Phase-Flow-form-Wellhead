#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supri-D Product 
Date: 6/11/2021

@author: Abdullah Alakeely, 
@ the code here is complementary for the, 
@thesis: Full-Field Analysis: A Machnine Learning Approach

Please cite the thesis when using the code.


# papers used this code

Alakeely, Abdullah , and Roland N. Horne. "Simulating the 
Behavior of Reservoirs with Convolutional and Recurrent 
Neural Networks." SPE Res Eval & Eng 23 (2020):
 0992–1005. doi: https://doi.org/10.2118/201193-PA


Alakeely, Abdullah A., and Roland N. Horne. "Application of Deep Learning 
Methods in Evaluating Well Production Potential Using Surface Measurements." Paper 
presented at the SPE Annual Technical Conference and Exhibition, Virtual, October 2020. 
doi: https://doi.org/10.2118/201785-MS

Contact : alakeeaa_at_stanford_dot_edu\
			or horne_at_stanford_dot_edu
"""

import numpy as np
from sklearn import preprocessing
import pandas as pd
import os
import keras
import datetime

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Conv1D, SimpleRNN, GRU
from keras.layers import TimeDistributed

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ["Times New Roman"]

class DataProcess:
    
    # input variables used in the experiment
    
    xinp= ['minutes','ChkSize' , 'WhP', 'WhT']
    
    # output variables used in the experiment
    
    yout =   ['QGas1av', 'QoStk1av','Qw1av']
    
    """ mapping experiment 
    
   Attributes
    ----------
    
    # class attributes
    
    input_variables : list, default xinp. The inputs used in this experiment        
    output_variables : list, default yout. The outputs used in this experiment        
    
    # inputs:
    training_case : scalar, default 1. The case number used in this experiment.
        

    
    
    """
    
    
    def __init__(self,  days = 365):
        
        
        # days during training
        
        self.days = days
        
        # input variables
        
        self.input_variables = DataProcess.xinp
        
        # output variables

        self.output_variables = DataProcess.yout
        
        # input variables size

        self.n_x = len(self.input_variables)
        
        
        # output variables size
        
        self.n_y = len(self.output_variables)
        
        self.case = self.load_well_data()
        
       
        #  grab input data 
        self.x_data = self.extract_variables(self.input_variables)
        
        #  grab output data 

        self.y_data = self.extract_variables(self.output_variables)
        
        #  perform normalization of input data 

        self.X, self.input_scaling_info = self.get_scaled(self.x_data)
        
        #  perform normalization of output data 

        self.y, self.output_scaling_info = self.get_scaled(self.y_data)

    def load_well_data(self):
        """
        # change this to load new data
    
        data is loaded per well
        """
        import pandas as pd
    
        # import the production data
        raw = pd.read_csv('data/l_condensate.csv')

        variables = ['minutes']+[list(raw.iloc[4].items())[x][1] for x in range(1,len(list(raw.iloc[4].items())))]
        data = pd.DataFrame()

        for col in range(len(list(raw.columns))):
            data[variables[col]] = raw[raw.columns[col]][6:]
        
        data['minutes'] = [x+1 for x in range(1377)]
    
        return data 
    

    def extract_variables(self,variables, start = 0, end = 1370):
        
        # function that extract desired variables defined in variables
        trainx = np.zeros((end, len(variables)))
        
        for i in range(len(variables)):
            temp = np.reshape(self.case[variables[i]][start:start+end], -1)
            trainx[:,i] = np.array(temp)
        return trainx
    
    def get_scaled(self,matrix):
        
        # prepare an empty array to collect the scaled vectors
        
        X_t = np.empty((1, matrix.shape[-2], matrix.shape[-1]))
        
        # prepare an temporary variable 

        scaling_ = []
        temp1 = np.zeros(matrix.shape)
    
        for colm in range(temp1.shape[-1]):
            
            scalerx = preprocessing.MinMaxScaler().fit(matrix[:,colm:colm+1])
            
            temp1[:,colm:colm+1] = scalerx.transform(matrix[:,colm:colm+1])
            scaling_.append(scalerx)

        X_t[:,] = temp1
        
        return X_t, scaling_
    
        
    def inverse_scaled(self, matrix, scaling_information):
        
        # prepare an empty array to collect the scaled vectors
        
        X_t = np.empty((1, matrix.shape[-2], matrix.shape[-1]))
        
        # prepare an temporary 

        
        temp1 = np.zeros(matrix.shape)
    
        for colm in range(temp1.shape[-1]):
            
            
            temp1[:,colm:colm+1] = scaling_information[colm].inverse_transform(matrix[:,colm:colm+1])
        

        X_t[:,] = temp1
        
        return X_t
    
    def reshape_to_three_dimensions(self,data):
        
        return np.reshape(data,(data.shape[0],1,data.shape[1]))
    
    # split dataset into train/test sets
    def split_dataset(self, X, y, split = True, three_dimensions = True,\
                      end_test = 30):
        
        

        
        
        train_span = self.days
        
        
        X_train = X[0][:train_span]
        y_train = y[0][:train_span]
        
        X_tes = X[0][:train_span+end_test]
        y_tes = y[0][:train_span+end_test]
        

        if three_dimensions == True:
            trainx = self.reshape_to_three_dimensions(X_train) 
            testx = self.reshape_to_three_dimensions(X_tes)
        else:
            trainx = X_train 
            testx = X_tes
        trainy = y_train
        testy = y_tes
        return trainx, trainy, testx, testy



class Models:
    
    def __init__(self ,x_train, y_train, n_h_1 = 4, n_h_2 = 4, activation1 = 'elu',\
                 activation2 = 'elu', cell_type = 'RNN',\
                 batch_size = 32, lr = 0.01):
        self.cell_type = cell_type
        self.n_h_1 = n_h_1
        self.n_h_2 = n_h_2
        self.activation1 = activation1
        self.activation2 = activation2
        self.batch_size = batch_size
        self.lr = lr
        
        
        if self.Algorithm == 'Algorithm II':
            n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[-1],\
            y_train.shape[-1]
            
            if self.cell_type == 'RNN':
                
                cell_t = SimpleRNN
            
            elif self.cell_type == 'GRU':
                
                cell_t = GRU
            
            elif self.cell_type == 'LSTM':
                
                cell_t = LSTM
            
            self.model = Sequential()
            self.model.add(TimeDistributed(Dense(self.n_h_1, activation='elu',input_shape=(n_timesteps,n_features)))) 

            self.model.add(cell_t(self.n_h_2, activation='elu',input_shape=(n_timesteps,n_features))) 
            self.model.add(Dense(n_outputs))

        

        elif self.Algorithm == 'Algorithm III':
            n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[-1],\
            y_train.shape[-1]
            
            if self.cell_type == 'RNN':
                
                cell_t = SimpleRNN
            
            elif self.cell_type == 'GRU':
                
                cell_t = GRU
            
            elif self.cell_type == 'LSTM':
                
                cell_t = LSTM
            
            self.model = Sequential()
            self.model.add(cell_t(self.n_h_1, activation=self.activation1,input_shape=(n_timesteps,n_features))) 
            self.model.add(Dense(self.n_h_2, activation=self.activation2,input_shape=(n_features,))) 

            self.model.add(Dense(n_outputs))



        elif self.Algorithm == 'Algorithm V':
            n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[-1],\
            y_train.shape[-1]
            
            if self.cell_type == 'RNN':
                
                cell_t = SimpleRNN
            
            elif self.cell_type == 'GRU':
                
                cell_t = GRU
            
            elif self.cell_type == 'LSTM':
                
                cell_t = LSTM
            
            self.model = Sequential()
            self.model.add(cell_t(self.n_h_1, activation=self.activation1,input_shape=(n_timesteps,n_features))) 

            self.model.add(Dense(n_outputs))
        elif self.Algorithm == 'Algorithm IV':
            n_timesteps, n_features, n_outputs = x_train.shape[0], x_train.shape[-1],\
            y_train.shape[-1]
            self.model = Sequential()
            self.model.add(Dense(self.n_h_1, activation=self.activation1,input_shape=(n_features,))) 
            self.model.add(Dense(n_outputs))


        elif self.Algorithm == 'Algorithm I':
            n_timesteps, n_features, n_outputs = x_train.shape[0], x_train.shape[-1],\
            y_train.shape[-1]
            self.model = Sequential()
            self.model.add(Dense(self.n_h_1, activation=self.activation1,input_shape=(n_features,))) 
            self.model.add(Dense(self.n_h_2, activation=self.activation2)) 
            self.model.add(Dense(n_outputs))



        elif self.Algorithm == 'Algorithm VI':
            n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[-1],\
            y_train.shape[-1]
            
            if self.cell_type == 'RNN':
                
                cell_t = SimpleRNN
            
            elif self.cell_type == 'GRU':
                
                cell_t = GRU
            
            elif self.cell_type == 'LSTM':
                
                cell_t = LSTM
            
            self.model = Sequential()
            self.model.add(cell_t(self.n_h_1,return_sequences = True, activation='elu',input_shape=(n_timesteps,n_features))) 
            self.model.add(cell_t(n_outputs, activation=self.activation1)) 
        
        
        elif self.Algorithm == 'Algorithm VII':
            n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[-1],\
            y_train.shape[-1]

            
            self.model = Sequential()
            self.model.add(Conv1D(padding = 'causal', filters = self.n_h_1,kernel_size = self.n_h_2,  activation=self.activation1,input_shape=(n_timesteps,n_features))) 
            self.model.add(Flatten())
            self.model.add(Dense(n_outputs))
        optimizer = keras.optimizers.Adam(lr=self.lr)
        
        self.model.compile(loss='mse', optimizer= optimizer)
        
        
    
    def train_model(self, epochs, verbose):
        
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=self.batch_size, verbose=verbose, validation_split=0.1)
    




class Trial(DataProcess, Models):
    
    def __init__(self, days = 365, n_h_1 = 4,\
                  n_h_2 = 4, activation1 = 'elu',activation2 = 'elu',cell_type = 'RNN', batch_size = 32,\
                     lr = 0.01, end_test = 30, results_directory = 'Experiment1',\
                         file_name = 'trial', Algorithms = 'Algorihtm I'):
        
        try:
            os.makedirs(results_directory)
        except:
            pass

        try:
            self.results_df = pd.read_pickle(os.path.join(results_directory,file_name+'.pkl'))

        except:
    
            self.results_df = pd.DataFrame()
        
        self.results_df['trial'] = len(self.results_df)+1
        
        
        self.Algorithm = Algorithms
        
        
        DataProcess.__init__(self, days)
        
        
        if self.Algorithm =='Algorithm VI' \
                or self.Algorithm =='Algorithm V' \
                    or self.Algorithm =='Algorithm VII' \
                        or self.Algorithm =='Algorithm II'\
                            or self.Algorithm == 'Algorithm III':

            self.data = self.split_dataset(self.X, self.y, end_test = end_test)
        elif self.Algorithm =='Algorithm I' or self.Algorithm == 'Algorithm IV':# or 'Algorithm VI' or 'Algorithm V' or 'Algorithm VII' :
            self.data = self.split_dataset(self.X, self.y, three_dimensions = False,end_test = end_test)
        
        self.x_train = self.data[0]
        self.y_train = self.data[1]
        self.x_test = self.data[2]
        self.y_test = self.data[3]
        
        
        Models.__init__(self, self.x_train, self.y_train,n_h_1,n_h_2,activation1, activation2, cell_type, batch_size, lr)
        
        self.training_loss_hist = []
        self.val_loss_hist = []
    
    def train(self,verbose = 0, epochs = 10):
        
        self.train_model(epochs, verbose)
        self.training_loss_hist.extend(self.model.history.history['loss'])
        self.val_loss_hist.extend(self.model.history.history['val_loss'])

        
    def get_prediction(self, results_directory, file_name,plot_results = True, return_scores = True, record_predictions = True):
        
        self.yhat = self.inverse_scaled(self.model.predict(self.x_test), self.output_scaling_info)[0]
        
        self.ytrue = self.y_data[:len(self.yhat)]
        
        if record_predictions:
            
            try:
                self.prediction_df = pd.read_pickle(os.path.join(results_directory,file_name+'_predictions.pkl'))
    
            except:
        
                self.prediction_df = pd.DataFrame()
            
            new_seg = {'index':np.transpose([x+1 for x in range(len(self.ytrue))]),
                       'True':np.transpose(self.ytrue),
                       'predicted':np.transpose(self.yhat),
                    }
            self.prediction_df = self.prediction_df.append(new_seg, ignore_index=True)
            
            
            try:
                predictions_f = os.path.join(results_directory,'predictions')
                os.makedirs(predictions_f)
            except:
                pass
            
            
            
            
            self.prediction_df.to_excel(os.path.join(predictions_f,file_name+'_predictions.xlsx'))
            self.prediction_df.to_pickle(os.path.join(predictions_f,file_name+'_predictions.pkl'))

        if plot_results:
            self.show_plot()
            
        if return_scores:
            return self.get_scores()


        

        
    
    def get_scores(self):
        train_score = self.prediction_score(self.ytrue[:len(self.y_train)],self.yhat[:len(self.y_train)], metric_to_use = 'r2')
        test_score = self.prediction_score(self.ytrue[len(self.y_train):len(self.yhat)],self.yhat[len(self.x_train):len(self.yhat)], metric_to_use = 'r2')
        score = self.prediction_score(self.ytrue,self.yhat, metric_to_use = 'r2')
        return train_score, test_score, score
    
    def show_plot(self):
        plt.figure(figsize = [10,6])
        self.prediction_visual_inspection(self.ytrue, self.yhat, self.days)
        
        
        
        
    def prediction_visual_inspection(self,true_signal, predicted_signal,days,\
                                  r = 2, c = 2, font_s = 14,\
                                 ):
        
        
        style = {'linestyle':'',
                 'marker':'o',
                 'markersize':2,
            }
        
        
        colors = ['r','g','b']
        number_of_plots = true_signal.shape[-1]
        y_limits = [true_signal.min()*0.95,true_signal.max()*1.05]
        x_limits = [-len(true_signal)*.05,len(true_signal)*1.05]
        
        for x in range(number_of_plots):
            plt.subplot(r,c,x+1)
            plt.plot(true_signal[:,x], color = 'gray', label = '$y$', **style)
            plt.plot(predicted_signal[:,x],color = colors[x], alpha = 0.6, label = '$\hat{y}$',**style)
            y_limits = [true_signal[:,x].min()*0.95,true_signal[:,x].max()*1.05]
            x_limits = [-len(true_signal[:,x])*.05,len(true_signal[:,x])*1.05]
            plt.ylim(y_limits[0], y_limits[1])
            plt.xlim(x_limits[0], x_limits[1])
        
            plt.xlabel('Time', fontsize = font_s)
            plt.ylabel('bbl/d', fontsize = font_s)
            plt.xticks(fontsize = font_s-2)
            plt.yticks(fontsize = font_s-2)
    
            plt.legend(fontsize = font_s-2)
            plt.axvline(days, linestyle = '--', color = 'k', alpha = 0.5, linewidth = 2)
        plt.tight_layout()
        
    def prediction_score(self, true_signal, predicted_signal, metric_to_use = 'mse'):

        number_of_curves = true_signal.shape[-1]

        if metric_to_use == 'mse':
            try:
                
                scores_list = [mean_squared_error(true_signal[:,x],predicted_signal[:,x]) for x in range(number_of_curves)]
            except:
                pass
            
            return scores_list
        
        if metric_to_use == 'mae':
            try:
                scores_list = [mean_absolute_error(true_signal[:,x],predicted_signal[:,x]) for x in range(number_of_curves)]
            except:
                pass
            return scores_list
        
        if metric_to_use == 'r2':
            try:
                scores_list = [r2_score(true_signal[:,x],predicted_signal[:,x]) for x in range(number_of_curves)]
            except:
                pass
            return scores_list
            
    def learning_behavior(self, font_s = 14):
        colors = ['r','b']
        y_limits = [-.01,0.2]
        x_limits = [-len(self.training_loss_hist)*.05,len(self.training_loss_hist)*1.05]
        
        plt.plot([x+1 for x in range(len(self.training_loss_hist))],self.training_loss_hist, color = colors[0], label = 'loss')
        plt.plot([x+1 for x in range(len(self.val_loss_hist))],self.val_loss_hist,color = colors[1], alpha = 0.4, label = 'val')
    
        plt.ylim(y_limits[0], y_limits[1])
        plt.xlim(x_limits[0], x_limits[1])
    
        plt.xlabel('epochs', fontsize = font_s)
        plt.ylabel('Mean Squared Error', fontsize = font_s)
        plt.xticks(fontsize = font_s-2)
        plt.yticks(fontsize = font_s-2)

        plt.legend(fontsize = font_s-2)
        plt.tight_layout()
        


def run_one_trial(xinp,
                  yout,
                  select,
                  days,
                  neurons1,
                  neurons2,
                  activation1,
                  activation2,
                  batch_size,
                  learning_rate,
                  test_span,
                  file_name,
                  results_directory,
                  epochs,
                  stops,
                  verbose,
                  plot_results,
                  record_predictions):
    
    algorithms_list = ['Algorithm I',    # feed-forward -> feed-forward -> feed-forward
                       'Algorithm II',   # feed-forward -> recurrent -> feed-forward
                       'Algorithm III',  # recurrent -> feed-forward -> feed-forward
                       'Algorithm IV',   # feed-forward -> feed-forward 
                       'Algorithm VI',   # recurrent -> recurrent
                       'Algorithm V',    # recurrent -> feed-forward
                       'Algorithm VII']  # Conv. 1D --> feed-forward
    
    
    
    # change cell type 
    cell_types = [None,   # no recurrent 
                  'RNN',  # choose from [RNN, LSTM, GRU]
                  'GRU',  # choose from [RNN, LSTM, GRU]
                  None,   # no recurrent 
                  'LSTM', # choose from [RNN, LSTM, GRU]
                  'LSTM', # choose from [RNN, LSTM, GRU]
                  None]   # no recurrent

    # paramers of the run          
    startTime = datetime.datetime.now()
         

    params = {
              
              'days' : days,
              'n_h_1' : neurons1,
              'n_h_2' : neurons2,
              'activation1' : activation1,
              'activation2' : activation2,
              'cell_type' : cell_types[select-1],
              'batch_size' : batch_size,
              'lr':learning_rate,
              'end_test':test_span,
              'file_name':file_name,
              'results_directory':results_directory,
              'Algorithms':algorithms_list[select-1]
              }
    
    params2 = {'epochs':epochs,
        'verbose':verbose}
    
    
    
    plt.figure()
    try_combination = Trial(**params)
        # input variables used in the experiment
    
    try_combination.xinp = xinp
    try_combination.yout = yout

    
    # output variables used in the experiment
    
    yout =   ['QGas1av', 'QoStk1av','Qw1av']
    
    for stop in range(stops):
        
        try_combination.train(**params2)
        
        # record and plot last training effort
        if stop == stops-1:
            plot_results = True
            record_predictions = True
    
        # get scores
        scores = try_combination.get_prediction(results_directory, file_name,\
                                  plot_results = plot_results, record_predictions = record_predictions)
        
        temp = {'training_score%s'%str(stop+1):scores[0],
                'testing_score%s'%str(stop+1):scores[1],
                'curve_score%s'%str(stop+1):scores[2],
                'mean_training_score%s'%str(stop+1):np.mean(scores[0]),
                'mean_testing_score%s'%str(stop+1):np.mean(scores[1]),
                'mean_curve_score%s'%str(stop+1):np.mean(scores[2])
            }
        params.update(temp,join='left')
    
    
    
    
    params.update(params2)
    
    params3 = {'inputs':xinp,
        'outputs':yout}
    
    params.update(params3)
    
    plt.subplot(2,2,4)
    try_combination.learning_behavior()
    
    try:
        plots_f = os.path.join(results_directory,'plots')
        os.makedirs(plots_f)
    except:
        pass
    plt.savefig(os.path.join(plots_f,file_name+'_%s.pdf'%str(len(try_combination.results_df)+1)))
    plt.show()
    
    
    
    try_combination.results_df['timeDiff'] = datetime.datetime.now() - startTime
    
    
    try_combination.results_df = try_combination.results_df.append(params, ignore_index=True)
    
    # save 
    try:
        scores_f = os.path.join(results_directory,'scores')
        os.makedirs(scores_f)
    except:
        pass
    try_combination.results_df.to_pickle(os.path.join(scores_f,file_name+'_scores.pkl'))
    try_combination.results_df.to_csv(os.path.join(scores_f,file_name+'_scores.csv'))
    try_combination.results_df.to_excel(os.path.join(scores_f,file_name+'_scores.xlsx'))
    
    
    print('R2 on training data = ', np.nanmean(scores[0]))
    print('R2 on testing data = ', np.nanmean(scores[1]))
    print('R2 on curve  = ', np.nanmean(scores[2]))



    print('train : on Gas = {0:0.2f},  Oil = {1:0.2f}, and Water = {2:0.2f}'.format(*scores[0]))
    print('test : on Gas = {0:0.2f},  Oil = {1:0.2f}, and Water = {2:0.2f}'.format(*scores[1]))
    print('curve : score on Gas = {0:0.2f},  Oil = {1:0.2f}, and Water = {2:0.2f}'.format(*scores[2]))


