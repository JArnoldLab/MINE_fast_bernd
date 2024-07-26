import middle_layer_super_controller
import multiprocessing
import sys
import os





for index , line in enumerate(open("best_model_each_allele_one_hot")):

    print(line.split(",")[0])
    allele = line.split(",")[0]
    file_directory = "new_data_mhcI"
    number_folds = 5
    threshold = 500
    epochs = int(line.split(",")[1])
    batch_size = int(line.split(",")[2])
    learning_rate = float(line.split(",")[3])
    activation_function = line.split(",")[4]
    number_neurons = int(line.split(",")[5])
    gd = line.split(",")[6]
    loss_function = line.split(",")[7].strip()

    middle_layer_super_controller.get_run_auc(file_directory, allele , number_folds, threshold, int(epochs), int(batch_size), float(learning_rate), activation_function, int(number_neurons), gd, loss_function)
