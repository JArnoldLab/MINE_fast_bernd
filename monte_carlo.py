import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.integrate import quad
import os
import random
import time
#from numba import cuda

plt.style.use('ggplot')


#------------------------------------------------------gpu functions-------------------------------------------------------------------

#@cuda.jit
#def hamiltonian_func_gpu(beta , gamma , data_values , y_values , variance_rec , prediction , hamiltonian_result):
#    (row , col) = cuda.grid(2)
#    g = cuda.cg.this_grid()
#    if row < data_values.shape[0] and col < data_values.shape[1]:
#        prediction[row,0] = prediction[row,0] + ((data_values[row,col] - gamma[row,0]) * beta[col,0])
#        g.sync()

#    row2 = cuda.grid(1)
#    if row2 < prediction.shape[0]:
#        hamiltonian_result[0] = hamiltonian_result[0] + (((y_values[row,0] - prediction[row,0]) * (y_values[row,0] - prediction[row,0])) * variance_rec[row,0])

#-------------------------------------------------------end gpu functions------------------------------------------------------------------------------




def hamiltonian_func(beta , gamma , data_values , y_values , variance):
    prediction = np.dot(data_values , beta).reshape(len(data_values) , 1)
    fraction = variance * (np.square(y_values - prediction))
    sum_experiments = np.sum(fraction)

    return (sum_experiments , prediction)

def hamiltonian_func_fast(delta_beta , data_values , y_values , y_previous , variance_rec , beta_selected):
    delta_y = (data_values[:,beta_selected] * delta_beta).reshape(y_values.shape)
    y_new = y_previous + delta_y
    hamiltonian = np.sum(np.square(y_values - y_new) * variance_rec)
    return(hamiltonian , y_new)




def ensemble(delta_hamiltonian):
    return np.exp(-delta_hamiltonian / 2)


def mc_update(beta , gamma , hamiltonian , y_previous , number_betas , step_width , data_values , y_values , variance , max_beta , min_beta):
    acceptance = 0
    beta_new = beta.copy()
    gamma_new = gamma.copy()
    len_theta = len(beta)# + len(gamma)
    theta_selected = random.randint(0, len_theta - 1)
    #if theta_selected < len(beta):
    beta_new[theta_selected] = beta_new[theta_selected] + (step_width * ((2 * np.random.uniform(0, 1)) - 1))  # in the general case I would alter on the chosen variable
    delta_beta = beta_new[theta_selected] - beta[theta_selected]
    #    gamma_new = gamma
    #else:
    #    gamma_new[theta_selected - len(beta)] = gamma_new[theta_selected - len(beta)] = gamma_new[theta_selected - len(beta)] + (step_width * ((2 * np.random.uniform(0 , 1)) - 1))
    #    beta_new = beta
    #(hamiltonian_new , y_new) = hamiltonian_func(beta_new , gamma_new , data_values , y_values , variance)
    (hamiltonian_new , y_new) = hamiltonian_func_fast(delta_beta , data_values , y_values , y_previous , variance , theta_selected)
    #hamiltonian_new_gpu = 0
    #prediction_gpu = np.zeros(y_values.shape)
    #beta_dev = cuda.to_device(beta)
    #gamma_dev = cuda.to_device(gamma)
    #data_values_dev = cuda.to_device(data_values)
    #y_values_dev = cuda.to_device(y_values)
    #variance_rec_dev = cuda.to_device(variance)
    #prediction_gpu_dev = cuda.to_device(prediction_gpu)
    #hamiltonian_new_gpu_dev = cuda.to_device(hamiltonian_new_gpu)
    #threadsperblock = (16 , 16)
    #blockspergrid_x = int(math.ceil(data_values.shape[0] / threadsperblock[0]))
    #blockspergrid_y = int(math.ceil(data_values.shape[1] / threadsperblock[1]))
    #blockspergrid = (blockspergrid_x , blockspergrid_y)
    #hamiltonian_func_gpu[blockspergrid , threadsperblock](beta_dev , gamma_dev , data_values_dev , y_values_dev , variance_rec_dev , prediction_gpu_dev , hamiltonian_new_gpu_dev)
    #hamiltonian_new_gpu = hamiltonian_new_gpu_dev.copy_to_host()

    delta_hamiltonian = hamiltonian_new - hamiltonian

    if delta_hamiltonian < 0:
        ratio = 2.0
    else:
        ratio = ensemble(delta_hamiltonian)

    if ratio >= 1:
        acceptance = 1
    else:
        v = np.random.uniform(0, 1)
        if v < ratio:
            acceptance = 1

    #print(str(hamiltonian_new) + "," + str(hamiltonian_new_test) + "," + str(hamiltonian) + "," + str(hamiltonian_fast) + "," + str(acceptance))
    if acceptance == 1:
        return (beta_new , gamma_new , hamiltonian_new , y_new , acceptance)
    else:
        return (beta , gamma , hamiltonian , y_previous , acceptance)




def mc_simulation(number_replica , initial_beta , initial_gamma , data_values , y_values , variance , number_equilibration_sweeps , number_betas , step_width , max_beta , min_beta , number_theta_vectors , number_decorrelation_sweeps , name_simulation):
    np.random.seed(int(time.time()))
    beta_vector_matrix = []
    gamma_vector_matrix = []
    sweeps_number = 0
    acceptance_rate = 0
    file_output_beta = open(name_simulation + "_beta" , "w")
    file_output_gamma = open(name_simulation + "_gamma" , "w")
    #file_test = open("test_" + name_simulation , "w")
    for r in range(number_replica):
        #start of metropolis for each replica
        beta = initial_beta
        gamma = initial_gamma
        (hamiltonian , y_previous) = hamiltonian_func(beta , gamma, data_values , y_values , variance)
        #hamiltonian_gpu = 0
        #prediction_gpu = np.zeros(y_values.shape)
        #beta_dev = cuda.to_device(beta)
        #gamma_dev = cuda.to_device(gamma)
        #data_values_dev = cuda.to_device(data_values)
        #y_values_dev = cuda.to_device(y_values)
        #variance_rec_dev = cuda.to_device(variance)
        #prediction_gpu_dev = cuda.to_device(prediction_gpu)
        #hamiltonian_gpu_dev = cuda.to_device(hamiltonian_gpu)
        #threadsperblock = (16, 16)
        #blockspergrid_x = int(math.ceil(data_values.shape[0] / threadsperblock[0]))
        #blockspergrid_y = int(math.ceil(data_values.shape[1] / threadsperblock[1]))
        #blockspergrid = (blockspergrid_x, blockspergrid_y)
        #hamiltonian_func_gpu[blockspergrid, threadsperblock](beta_dev, gamma_dev, data_values_dev, y_values_dev, variance_rec_dev, prediction_gpu_dev, hamiltonian_gpu_dev)
        #hamiltonian_gpu = hamiltonian_gpu_dev.copy_to_host()
        file_output_plot_test_ham = open("data_plot_test_ham_" + str(r) + "_" + name_simulation, "w")
        file_output_plot_test_beta = open("data_plot_test_beta_" + str(r) + "_" + name_simulation , "w")
        for equil_sweeps in range(number_equilibration_sweeps):
            for theta_updaste in range(number_betas):
                (beta , gamma , hamiltonian , y_previous , acceptance) = mc_update(beta , gamma , hamiltonian , y_previous , number_betas , step_width , data_values , y_values , variance , max_beta , min_beta)
                acceptance_rate = acceptance_rate + acceptance
            #if equil_sweeps % 100 == 0:
            #    file_test.write(str(hamiltonian) + " " + str(float(hamiltonian_gpu)) + "\n")
            #----------------------------Plot stuff-------------------------------------------
            file_output_plot_test_ham.write(str(hamiltonian) + "\t" + str(sweeps_number) + "\n")
            for i in range(len(beta)):
                if i != len(beta) - 1:
                    file_output_plot_test_beta.write(str(beta[i]) + " ")
                else:
                    file_output_plot_test_beta.write(str(beta[i]) + "\t")
            file_output_plot_test_beta.write(str(sweeps_number) + "\n")

            sweeps_number = sweeps_number + 1
            #------------------------------------------------------------------------------------

        beta_vector_list = []
        gamma_vector_list = []
        for theta_acum in range(number_theta_vectors):
            for decorr_sweeps in range(number_decorrelation_sweeps):
                for theta_update in range(number_betas):
                    (beta , gamma , hamiltonian , y_previous , acceptance) = mc_update(beta , gamma , hamiltonian , y_previous , number_betas , step_width , data_values , y_values , variance , max_beta , min_beta)
                    acceptance_rate = acceptance_rate + acceptance

                #-----------------Plot stuff--------------------------------------------------------------
                file_output_plot_test_ham.write(str(hamiltonian) + "\t" + str(sweeps_number) + "\n")
                for i in range(len(beta)):
                    if i != len(beta) - 1:
                        file_output_plot_test_beta.write(str(beta[i]) + " ")
                    else:
                        file_output_plot_test_beta.write(str(beta[i]) + "\t")
                file_output_plot_test_beta.write(str(sweeps_number) + "\n")

                sweeps_number = sweeps_number + 1  # Plot stuff
                #-----------------------------------------------------------------------------------------

            beta_vector_list.append(beta)
            gamma_vector_list.append(gamma)
            for b_element in range(len(beta)):
                if b_element == len(beta) - 1:
                    file_output_beta.write(str(float(beta[b_element])))
                else:
                    file_output_beta.write(str(float(beta[b_element])) + " ")
            file_output_beta.write("|")
            for g_element in range(len(gamma)):
                if g_element == len(gamma) - 1:
                    file_output_gamma.write(str(float(gamma[g_element])))
                else:
                    file_output_gamma.write(str(float(gamma[g_element])) + " ")
            file_output_gamma.write("|")

            #--------Plot stuff-------------------------------------------------------------------------------------------------------------
            #file_output_plot_test.write(str(beta[0]) + "," + str(beta[1]) + "," + str(beta[2]) + "\t" + str(hamiltonian) + "\t" + str(sweeps_number) + "\n")
            #sweeps_number = sweeps_number + 1  # Plot stuff
            #---------------------------------------------------------------------------------------------------------------------------------

        beta_vector_matrix.append(beta_vector_list)
        gamma_vector_matrix.append(gamma_vector_list)
        file_output_beta.write("\n")
        file_output_gamma.write("\n")
        #file_test.close()
        file_output_plot_test_ham.close()
        file_output_plot_test_beta.close()

    file_output_beta.close()
    file_output_gamma.close()
    print(acceptance_rate / ((number_equilibration_sweeps * number_betas) + (number_theta_vectors * number_decorrelation_sweeps * number_betas)))

    return (np.array(beta_vector_matrix) , np.array(gamma_vector_matrix))


def read_file(name_file):
    theta_vector_matrix = []
    for index , line in enumerate(open(name_file)):
        list_theta_vectors = []
        list_theta_strings = line.split("|")
        for theta in list_theta_strings:
            theta_vector = theta.strip().split(" ")
            if theta_vector != ['']:
                theta_vector = list(map(lambda x: float(x) , theta_vector))
                list_theta_vectors.append(theta_vector)
        theta_vector_matrix.append(list_theta_vectors)
    return np.array(theta_vector_matrix)

def read_data(file_matrix):
    list_matrix = []
    for index , line in enumerate(open(file_matrix)):
        data = line.strip().split(" ")
        data = list(map(lambda x:float(x) , data))
        list_matrix.append(data)
    return np.array(list_matrix)

def read_test_values(test_values):
    list_y_values = []
    for index , line in enumerate(open(test_values)):
        data = line.strip()
        list_y_values.append(float(data))
    return np.array(list_y_values).reshape((len(list_y_values) , 1))

