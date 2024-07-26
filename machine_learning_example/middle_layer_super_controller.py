import os
import main_nn
import numpy as np


def read_examples(file_name , data_train , file_name_test):
    list_labels = []
    list_data = []
    list_aux = []
    number = 1000000
    number_test = 1000000

    for item in data_train.split(","):
        for index , line in enumerate(open(file_name + "/" + item)):
            temp = line.split()
            label = round(float(temp[0]) , 2)
            if len(temp[1:]) < number:
                number = len(temp[1:])

    for index , line in enumerate(open(file_name_test)):
        temp = line.split()
        label = round(float(temp[0]) , 2)
        if len(temp[1:]) < number_test:
            number_test = len(temp[1:])

    if number < number_test:
        number_features = number
    else:
        number_features = number_test

    for item in data_train.split(","):
        for index , line in enumerate(open(file_name + "/" + item)):
            temp = line.split()
            label = float(temp[0])
            list_labels.append(label)
            list_aux = []
            for element in temp[1:]:
                feature = float(element.split(":")[1])
                list_aux.append(feature)
            list_aux = list_aux[:number_features] #number of features fixed because of variable length features
            list_data.append(list_aux)

    data = np.matrix(list_data)
    labels = np.array(list_labels)

    list_data = []
    list_labels = []
    list_aux = []

    for index , line in enumerate(open(file_name_test)):
        temp = line.split()
        label = float(temp[0])
        list_labels.append(label)
        list_aux = []
        for element in temp[1:]:
            feature = float(element.split(":")[1])
            list_aux.append(feature)
        list_aux = list_aux[:number_features] #number of features fixed because of variable length features
        list_data.append(list_aux)

    data_test = np.matrix(list_data)
    labels_test = np.array(list_labels)

    return(data , labels , data_test , labels_test , number_features)


def get_run_auc(file_directory , allele , number_folds , threshold , epochs , batch_size , learning_rate , activation_function , number_neurons , gd , loss_function):


    if os.path.exists("results/" + file_directory + "/" + allele) == False:
        os.makedirs("results/" + file_directory + "/" + allele)


    auc = 0
    auc_temp = 0
    best_model = 0
    for i in range(number_folds):
        data_test = "set" + str(i)
        j = 0
        data_train = ""
        while j < number_folds:
            if j != i:
                data_train = data_train + "set" + str(j)
                if len(data_train.split(",")) != number_folds - 1:
                    data_train = data_train + ","
            j = j + 1

        (x, y, x_test, y_test, number_features) = read_examples("../2-convert_data_encode/results/" + file_directory + "/" + allele , data_train , "../2-convert_data_encode/results/" + file_directory + "/" + allele + "/" + data_test)
        try:
            (auc , true_positives , false_negatives , false_positives , true_negatives , data_size) = main_nn.prediction(i , file_directory , allele , x , y , x_test , y_test , number_features , threshold , epochs , batch_size , learning_rate , activation_function , number_neurons , gd , loss_function)
        except Exception as e:
            print(str(e))
