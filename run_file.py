#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supri-D Product 
Date: 6/11/2021

@author: Abdullah Alakeely, 
@ the code here is complementary for the, 
@thesis: Full-Field Analysis: A Machnine Learning Approach

Please cite the thesis when using the code.

Contact : alakeeaa_at_stanford_dot_edu\


"""


from supporting_modules import run_one_trial

""" mapping experiment 
    
   Attributes
    ----------
    
    # class attributes
    
    input_variables : list, default xinp. The inputs used in this experiment        
    output_variables : list, default yout. The outputs used in this experiment        
    
    # inputs:
    select : scalar, default 1. algorithm to use from the list below:
    
    Algorithm I : feed-forward -> feed-forward -> feed-forward
    Algorithm II : feed-forward -> recurrent -> feed-forward
    Algorithm III : recurrent -> feed-forward -> feed-forward
    Algorithm IV : feed-forward -> feed-forward
    Algorithm VI : recurrent --> recurrent
    Algorithm V : recurrent -> feed-forward
    Algorithm V : one-dimensional causal convolution -> feed-forward



    stops : scalar, default 2. number of stops to evaluate the performance.
    epochs: scalar, default 5. number of passes over all training examples.
    plot_results: logical. input to get prediction function, if true it plots the results 
    record_predictions: logical .input to get prediction function, if true it plots the results 

    
    
    
"""


# Only change here

xinp= ['minutes','ChkSize' , 'WhP', 'WhT']

# output variables used in the experiment

yout =   ['QGas1av', 'QoStk1av','Qw1av']

results_directory = 'Experiment1'

file_name = 'trial'

select = 7    # algorithm from list (max is 6)

stops = 2      # stops to evaluate performance

epochs = 5      # trainining epochs before stoping to evaluate

days = 500     # size of training in timesteps

neurons1 = 30    # size of hidden layer 1

neurons2 = 1    # size of hidden layer 2, or size of kernel when using 'Algorithm VII'

activation1 = 'relu'    # activation of hidden layer 1

activation2 = 'relu'   # activation of hidden layer 2

batch_size  =8  # batch size during training

learning_rate = 0.02 

test_span = 100     # test data set in the future (forcast days after training)

plot_results = False    # if true it plots the results 

record_predictions = False  # if true it will record predicion 

verobse = 0   # shows training progress if set to 1




# function to run one trial



run_one_trial(xinp,
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
                  verobse,
                  plot_results,
                  record_predictions)

