import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
import math





#--------------------------------------------------------------------------------------------------main part--------------------------------------------------------------------------------------





def prediction( ind , file_directory , allele , x , y , x_test , y_test , number_features , threshold , num_epochs , batch_size , learning_rate , activation_function , number_neurons , gd , loss_function):



    if activation_function == "relu":
        activation = torch.nn.ReLU()
    elif activation_function == "tanh":
        activation = torch.nn.Tanh()
    elif activation_function == "lrelu":
        activation = torch.nn.LeakyReLU()
    elif activation_function == "sigmoid":
        activation = torch.nn.Sigmoid()


    model = torch.nn.Sequential(
        torch.nn.Linear(number_features , number_neurons),
        activation,
        torch.nn.Linear(number_neurons , 1)
    )

#set cuda datatype and cuda mode if gpu is available
    if torch.cuda.is_available():
        ftype = torch.cuda.FloatTensor
        model.cuda()
    else:
        ftype = torch.FloatTensor




    if gd == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif gd == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif gd == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if loss_function == "mse":
        loss_func = torch.nn.MSELoss()
    elif loss_function == "mae":
        loss_func = torch.nn.L1Loss()
    elif loss_function == "huber":
        loss_func = torch.nn.SmoothL1Loss()
    #elif loss_function == "logcos":
    #elif loss_function == "quantile":
        #loss_func = torch.nn.Loss

    train_size = x.shape[0]
    test_size = x_test.shape[0]
# train the network
    for t in range(num_epochs):
        model.train()
        s = 0
        while s < train_size:
            e = min(s + batch_size , train_size)
            batch_x = x[s : e]
            batch_y = y[s : e]
            X = Variable(torch.from_numpy(batch_x).type(ftype) , requires_grad = False)
            Y = Variable(torch.from_numpy(batch_y).type(ftype) , requires_grad = False)
            Y = Y.view(-1 , 1)
            optimizer.zero_grad()
            prediction = model(X)
            loss = loss_func(prediction, Y)
            loss.backward()
            optimizer.step()
            s = e



#test

    torch.save(model.state_dict() , "results/" + file_directory + "/" + allele + "/model_" + str(ind) + ".pth")
    model.eval()
    X_test = Variable(torch.from_numpy(x_test).type(ftype) , requires_grad = False)
    Y_test = Variable(torch.from_numpy(y_test).type(ftype) , requires_grad = False)
    pred_Y = model(X_test)
    pred_Y,_ = torch.max(pred_Y , dim = 1)

    del model
    del X
    del Y
    del X_test
    if torch.cuda.is_available():
        pred_Y_result = pred_Y.cpu()
        Y_test_result = Y_test.cpu()
        torch.cuda.empty_cache()

    labels_pred = pred_Y.data.numpy()
    labels_test = Y_test.data.numpy()

#AUC

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    sensitivity = 0
    specificity = 0

    for i in range(len(labels_pred)):
        transform_labels_test = math.exp(math.log(50000) * (1 - labels_test[i]))
        transform_labels_pred = math.exp(math.log(50000) * (1 - labels_pred[i]))
        if transform_labels_test <= float(threshold) and transform_labels_pred <= float(threshold):
            true_positives = true_positives + 1
        elif transform_labels_test <= float(threshold) and transform_labels_pred > float(threshold):
            false_negatives = false_negatives + 1
        elif transform_labels_test > float(threshold) and transform_labels_pred <= float(threshold):
            false_positives = false_positives + 1
        elif transform_labels_test > float(threshold) and transform_labels_pred > float(threshold):
            true_negatives = true_negatives + 1
    if (true_positives + false_negatives) > 0:
        sensitivity = true_positives / (true_positives + false_negatives)
    if (true_negatives + false_positives) > 0:
        specificity = true_negatives / (false_positives + true_negatives)


    auc = (sensitivity + specificity) / 2
    return(auc , true_positives , false_negatives , false_positives , true_negatives , (train_size + test_size))




#file = sys.argv[1]
#file_test = sys.argv[2]
#file_directory = sys.argv[3]
#threshold = sys.argv[4]

#auc = prediction(file , file_test , file_directory , threshold , 300 , 100 , 0.2)

#print(auc)
