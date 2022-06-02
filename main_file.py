import numpy as np
import monte_carlo as mc
import matrix_generation as mg
import multiprocessing
import plot as plt




#parameters of the simulation

name_simulation = "run_26"
file_matrix = "matrix_final_mine"
labels_file = "labels"
number_replica = 1
step_width = 4.0
number_equilibration_sweeps = 40000
number_decorrelation_sweeps = 100
number_theta_vectors = 1000
data_values = mc.read_data(file_matrix)
(number_datapoints , number_betas) = data_values.shape
max_beta = 1.0
min_beta = 0.0
beta = [0.0] * number_betas
gamma = [0.0] * number_datapoints
y_values = mc.read_test_values("dry_weight_mine")#np.random.normal(0 , 1 , (number_datapoints , 1))
variance = mc.read_test_values("variance_mine")#np.var(data_values , axis = 1).reshape(number_datapoints , 1)
variance_rec = 1/variance
for i in range(len(variance_rec)):
    if np.isinf(variance_rec[i]):
        variance_rec[i] = 0
initial_beta = np.random.uniform(0 , 1 , (number_betas , 1)) #* (max_beta - min_beta) + min_beta
initial_gamma = np.random.uniform(0 , 1 , (number_datapoints , 1))
number_bins = 8
binwidth = (2 * max_beta) / number_bins
s_cut_evals = 10000
number_new_experiments = 81
number_max_elements_per_partition = 2
number_max_elements_initial = 3
number_parallel_process_mine = 5
criteria_option = 1 # 1 for e_matrix , 2 for d_matrix
option_wf = 3 # 1 for simulation, storage and MINE , 2 for simulation and storage , 3 for MINE from a file



if option_wf == 1:
    (beta_vector_matrix , gamma_vector_matrix) = mc.mc_simulation(number_replica , initial_beta , initial_gamma , data_values , y_values , variance_rec , number_equilibration_sweeps , number_betas , step_width , max_beta , min_beta , number_theta_vectors , number_decorrelation_sweeps , name_simulation)
    theta_vector_matrix_avg = mg.average_theta_matrix(beta_vector_matrix)
    (new_data_recommended_d, new_data_recommended_e) = mg.mine_criteria(theta_vector_matrix_avg, s_cut_evals, number_new_experiments)
elif option_wf == 2:
    (theta_vector_matrix , gamma_vector_matrix) = mc.mc_simulation(number_replica , initial_beta , initial_gamma , data_values , y_values , variance_rec , number_equilibration_sweeps , number_betas , step_width , max_beta , min_beta , number_theta_vectors , number_decorrelation_sweeps , name_simulation)
elif option_wf == 3:
    beta_vector_matrix = mc.read_file(name_simulation + "_beta")
    gamma_vector_matrix = mc.read_file(name_simulation + "_gamma")
    beta_vector_matrix_avg = mg.average_theta_matrix(beta_vector_matrix)
    gamma_vector_matrix_avg = mg.average_theta_matrix(gamma_vector_matrix)
    (labels , data_vectors) = mg.get_u_data(labels_file , file_matrix)
    mg.mine_criteria_multiprocessing(beta_vector_matrix_avg , gamma_vector_matrix_avg , labels , data_vectors , number_datapoints , s_cut_evals , number_max_elements_per_partition , name_simulation , number_new_experiments , criteria_option , number_max_elements_initial)
    #plt.do_plots(name_simulation , number_replica - 1 , 200 , number_equilibration_sweeps)



