import numpy as np
import operator
import itertools
import scipy
import math
import multiprocessing
import time


def calculate_expected(g_function):
    g_function_sum = np.sum(g_function)
    return g_function_sum / len(g_function)


def calculate_expected_two_values(g_function_1 , g_function_2):
    g_function_mult = g_function_1 * g_function_2
    g_function_sum = np.sum(g_function_mult)
    return g_function_sum / len(g_function_mult)


def d_matrix_creation(beta_matrix , gamma_matrix , data_values , data_values_accumulate , data_vectors , data_vectors_old , number_new_experiments):
    d_matrix = np.empty(((number_new_experiments + len(data_values_accumulate)) , (number_new_experiments + len(data_values_accumulate))))
    counter_u = 0
    g_function_list = []

    for i in range(len(data_values_accumulate)):
        g_function = np.dot(beta_matrix , data_vectors_old[data_values_accumulate[i]])
        g_function_list.append(g_function)

    for i in range(number_new_experiments):
        g_function = np.dot(beta_matrix , data_vectors[data_values[i]])
        g_function_list.append(g_function)

    for i in range((number_new_experiments + len(data_values_accumulate))):
        for j in range((number_new_experiments + len(data_values_accumulate))):
            experiment = i
            experiment_prime = j
            g_function_i = g_function_list[experiment]
            g_function_j = g_function_list[experiment_prime]

            expected_two_values = calculate_expected_two_values(g_function_i , g_function_j)
            expected_one = calculate_expected(g_function_i)
            expected_two = calculate_expected(g_function_j)
            value = expected_two_values - (expected_one * expected_two)
            d_matrix[i][j] = value
    return d_matrix


def e_matrix_creation(d_matrix):
    (rows_number , cols_number) = d_matrix.shape
    e_matrix = np.empty((rows_number , cols_number))
    for i in range(rows_number):
        for j in range(cols_number):
            if d_matrix[i][i] != 0 and d_matrix[j][j] != 0:
                e_matrix[i][j] = d_matrix[i][j] / (np.sqrt(d_matrix[i][i]) * np.sqrt(d_matrix[j][j]))
            else:
                e_matrix[i][j] = 0

    return e_matrix

def average_theta_matrix(theta_matrix):
    (number_replica , number_theta_vectors_acum , number_theta_parameters) = theta_matrix.shape
    matrix_avg = np.zeros((number_theta_vectors_acum , number_theta_parameters))
    for i in range(number_replica):
        matrix_avg = matrix_avg + theta_matrix[i]
    matrix_avg = matrix_avg / number_replica
    return matrix_avg

def regularize_evals(evals , s_cut):
    min_value = evals[0] / s_cut
    for i in range(len(evals)):
        evals[i] = max(evals[i] , min_value)
    return evals


def determinant_squared_diag(eivals):
    log_eivals = np.log10(eivals)
    determinant = np.sum(log_eivals)
    return determinant

def print_d_matrix_e_format(d_matrix):
    file_output = open("d_matrix.csv", "w")
    file_output.write("index_row,index_col,element_matrix\n")
    for j in range(len(d_matrix[0])):
        for i in range(len(d_matrix)):
            file_output.write(str(i) + "," + str(j) + "," + str(d_matrix[i][j]) + "\n")
    file_output.close()

def print_evals_e_format(evals):
    evals = sorted(evals , reverse = True)
    file_output = open("evals.csv" , "w")
    file_output.write("index,elemennt_array\n")
    for i in range(len(evals)):
        file_output.write(str(i) + "," + str(evals[i]) + "\n")
    file_output.close()

def matrix_verification(d_matrix , evals , evecs):
    list_difference = []
    max_eigval = max(abs(evals))
    for i in range(len(evecs[0])):
        matrix_part = np.matmul(d_matrix , evecs[:,i])
        evecs_evals_part = evals[i] * evecs[:,i]
        difference = matrix_part - evecs_evals_part
        rel_err = np.linalg.norm(difference) / (max_eigval * np.linalg.norm(evecs[:,i]))
        list_difference.append(rel_err)
    return list_difference

def read_bernd_file_monte_carlo(bernd_file_montecarlo):
    list_theta = []
    list_theta_vector = []
    for index , line in enumerate(open(bernd_file_montecarlo)):
        list_theta = []
        if index != 0:
            data = line.split(",")
            list_theta.append(float(data[3].strip()))
            list_theta.append(float(data[4].strip()))
            list_theta.append(float(data[5].strip()))
            list_theta_vector.append(list_theta)
    return np.array(list_theta_vector)

def read_bernd_file_u_vectors(bernd_file_u_vectors):
    list_experiments = []
    list_experiment_1 = []
    list_experiment_2 = []
    bernd_results = {}
    for index , line in enumerate(open(bernd_file_u_vectors)):
        list_experiment_1 = []
        list_experiment_2 = []
        if index != 0:
            data = line.split(",")
            list_experiment_1.append(float(data[3].strip()))
            list_experiment_1.append(float(data[4].strip()))
            list_experiment_1.append(float(data[5].strip()))
            list_experiment_2.append(float(data[6].strip()))
            list_experiment_2.append(float(data[7].strip()))
            list_experiment_2.append(float(data[8].strip()))
            list_experiments.append((list_experiment_1 , list_experiment_2))
            bernd_results[str(list_experiment_1[0]) + "," + str(list_experiment_1[1]) + "," + str(list_experiment_1[2]) + "," + str(list_experiment_2[0]) + "," + str(list_experiment_2[1]) + "," + str(list_experiment_2[2])] = float(data[1].strip())

    return (list_experiments , bernd_results)



def mine_criteria(beta_vector_matrix_avg , gamma_vector_matrix_avg , data_values_range , labels , labels_index_accumulate , data_vectors , data_vectors_old , number_datapoints , s_cut_evals , number_new_experiments , name_simulation):
    list_values_det = []
    list_values_det_e = []
    #for i in range(len(data_values_range)):
    d_matrix_u_set = d_matrix_creation(beta_vector_matrix_avg , gamma_vector_matrix_avg , data_values_range , labels_index_accumulate , data_vectors , data_vectors_old , number_new_experiments)
    e_matrix_u_set = e_matrix_creation(d_matrix_u_set)
    evals_d_matrix = sorted(scipy.linalg.eigh(d_matrix_u_set , eigvals_only = True) , reverse = True)
    evals_e_matrix = sorted(scipy.linalg.eigh(e_matrix_u_set , eigvals_only = True) , reverse = True)

    evals_d_matrix = regularize_evals(evals_d_matrix, s_cut_evals)
    evals_e_matrix = regularize_evals(evals_e_matrix, s_cut_evals)
    result_d_matrix = determinant_squared_diag(evals_d_matrix)
    result_e_matrix = determinant_squared_diag(evals_e_matrix)
        #list_values_det.append((result_d_matrix , i))
        #list_values_det_e.append((result_e_matrix , i))

    #list_values_det = sorted(list_values_det , key = operator.itemgetter(0) , reverse = True)
    #list_values_det_e = sorted(list_values_det_e , key = operator.itemgetter(0) , reverse = True)
    #print(data_values_range)
    #print("\n")
    #print("New subset")
    file_output = open(name_simulation + "_labels_results" , "a")

    for i in range(len(data_values_range)):
        file_output.write(labels[data_values_range[i]] + " ")
    file_output.write("|" + str(result_d_matrix) + "|" + str(result_e_matrix) + "\n")
    file_output.close()




def get_u_data(labels_file , file_matrix):
    list_labels = []
    list_matrix = []
    for index , line in enumerate(open(labels_file)):
        list_labels.append(line.strip())

    for index , line in enumerate(open(file_matrix)):
        data = line.strip().split(" ")
        data = list(map(lambda x: float(x) , data))
        list_matrix.append(data)

    return (list_labels , np.array(list_matrix))


def create_grid(data_vectors , number_max_partitions):
    list_indexes = list(range(len(data_vectors)))
    list_combinations = list(itertools.combinations(list_indexes , number_max_partitions))

    return list_combinations

def read_grid(file_grid_values , number_max_partitions):
    list_current_labels_index = []
    labels_current = []
    for index , line in enumerate(open(file_grid_values)):
        list_current_labels_index.append(int(line.strip().split("|")[0]))
        labels_current.append(line.strip().split("|")[1])
    list_combinations = list(itertools.combinations(list_current_labels_index , number_max_partitions))

    return (list_combinations , labels_current)


def get_labels_general_index(labels_included , labels):
    list_accumulate_index = []
    for i in range(len(labels_included)):
        index = labels.index(labels_included[i])
        list_accumulate_index.append(index)
    return list_accumulate_index



def mine_criteria_multiprocessing(beta_vector_matrix_avg , gamma_vector_matrix_avg , labels , data_vectors , number_datapoints , s_cut_evals , number_max_elements_per_partition , name_simulation , number_new_experiments , criteria_option , number_max_elements_initial):
    list_accumulate = []
    list_accumulate_index = []
    i = 0
    labels_current = labels.copy()
    data_vectors_current = data_vectors.copy()
    while i < number_new_experiments:
        if i == 0:
            data_values_range = create_grid(data_vectors_current, number_max_elements_initial)
            number_elements = number_max_elements_initial
        else:
            data_values_range = create_grid(data_vectors_current , number_max_elements_per_partition)
            number_elements = number_max_elements_per_partition

        file_output_labels = open(name_simulation + "_labels_results", "w")
        file_output_labels.write("Accessions|log(det) D-matrix|log(det) E-matrix\n")
        file_output_labels.close()
        for j in range(len(data_values_range)):
            mine_criteria(beta_vector_matrix_avg , gamma_vector_matrix_avg , data_values_range[j] , labels_current , list_accumulate_index , data_vectors_current , data_vectors , number_datapoints , s_cut_evals , number_elements , name_simulation)
        (labels_d , labels_e , log_det_d , log_det_e) = get_experiment_accessions(name_simulation)
        if criteria_option == 1:
            list_accumulate.extend(labels_e)
            list_index = get_labels_general_index(labels_e , labels)
            list_accumulate_index.extend(list_index)
            list_accumulate = list(set(list_accumulate))
            list_accumulate_index = list(set(list_accumulate_index))
            (data_vectors_current , labels_current , list_current_labels_indexes) = remove_labels_data(data_vectors_current , labels_current , labels_e)
        else:
            list_accumulate.extend(labels_d)
            list_index = get_labels_general_index(labels_d , labels)
            list_accumulate_index.extend(list_index)
            list_accumulate = list(set(list_accumulate))
            list_accumulate_index = list(set(list_accumulate_index))
            (data_vectors_current , labels_current , list_current_labels_indexes) = remove_labels_data(data_vectors_current , labels_current , labels_d)
        i = i + number_elements


    file_output_final_labels = open(name_simulation + "_final_accessions" , "w")
    for i in range(len(list_accumulate)):
        file_output_final_labels.write(list_accumulate + " ")
    file_output_final_labels.write("|" + str(log_det_d) + "|" + str(log_det_e))
    file_output_final_labels.close()


def remove_labels_data(data_vectors , labels , labels_remove):
    list_indexes = []
    for element in labels_remove:
        index_element = labels.index(element)
        list_indexes.append(index_element)
        del labels[index_element]
        data_vectors = np.delete(data_vectors , index_element , 0)

    list_current_indexes = list(range(len(labels)))
    return (data_vectors , labels , list_current_indexes)

def get_experiment_accessions(name_simulation):
    list_data_d_matrix = []
    list_data_e_matrix = []
    for index , line in enumerate(open(name_simulation + "_labels_results")):
        if index != 0:
            data = line.split("|")
            list_data_d_matrix.append((data[0] , float(data[1])))
            list_data_e_matrix.append((data[0] , float(data[2].strip())))
    list_data_d_matrix = sorted(list_data_d_matrix , key = operator.itemgetter(1) , reverse = True)
    list_data_e_matrix = sorted(list_data_e_matrix , key = operator.itemgetter(1) , reverse = True)

    return(list_data_d_matrix[0][0].strip().split(" ") , list_data_e_matrix[0][0].strip().split(" ") , list_data_d_matrix[0][1] , list_data_e_matrix[0][1])


