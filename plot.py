import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.integrate import quad
import os
import random
import time
import operator

plt.style.use('ggplot')





def read_data_plot(file_name):
    theta_matrix = []
    list_sweeps = []
    for index , line in enumerate(open(file_name)):
        data = line.split("\t")
        theta_vector = data[0].split(",")
        theta_vector = list(map(lambda x: float(x.strip()) , theta_vector))
        theta_matrix.append(theta_vector)
        list_sweeps.append(int(data[2].strip()))
    return (np.array(theta_matrix) , list_sweeps)

def contour_plot(data_values , recommended_data_d):
    x = []
    y = []
    z = []
    log_det = []
    triangles_alt = []
    for i in range(len(recommended_data_d)):
        (x_comp , y_comp , z_comp) = data_values[recommended_data_d[i][1]][0]
        log_det_comp = recommended_data_d[i][0]
        x.append(x_comp)
        y.append(y_comp)
        z.append(z_comp)
        log_det.append(log_det_comp)
    triangles = mtri.Triangulation(x , y)
    plt.tricontourf(triangles , log_det)
    plt.xlabel("u1 coordinate")
    plt.ylabel("u2 coordinate")
    plt.title("Contour plot based on u vector coordinates and volume value")
    plt.show()


def plot_chi_sq_sweeps(file , number_equilibration_sweeps , name_simulation):
    sweeps = []
    chi_sq = []
    for index , line in enumerate(open(file)):
        data = line.split("\t")
        chi_sq.append(math.log10(float(data[0])))
        sweeps.append(int(data[1].strip()))

    plt.figure()
    plt.plot(sweeps , chi_sq)
    plt.axvline(x = number_equilibration_sweeps , color = 'blue')

    #plt.xlabel("Eql sweeps: 1 to " + str(number_equilibration_sweeps) + " , Acc: " + str(number_equilibration_sweeps) + " to " + str(len(sweeps)))
    plt.xlabel("Sweeps")
    plt.ylabel("Log(Chi-Squared)")
    plt.ylim(min(chi_sq) , max(chi_sq))
    plt.title("Trajectory Chi-Squared in Eq and Acc stages")
    plt.savefig("chi-squared_" + name_simulation , bbox_inches = 'tight')
    plt.show()

def plot_betas_sweeps(file , number_equilibration_sweeps , name_simulation):
    betas = []
    sweeps = []
    for index , line in enumerate(open(file)):
        data = line.split("\t")
        data_betas = data[0].split(" ")
        data_betas = list(map(lambda x: float(x[1 : len(x) - 1]) , data_betas))
        betas.append(data_betas)
        sweeps.append(int(data[1].strip()))

    plt.figure()
    plt.plot(sweeps , betas)
    plt.xlabel("Equilibration sweeps")
    plt.ylabel("Beta parameters")
    plt.title("Beta parameters at every equilibration sweep")
    plt.savefig("betas_sweeps_" + name_simulation , bbox_inches = 'tight')
    plt.show()




def plot_log_chi_sq_sweeps(chi_sq , sweeps , number_equilibration_sweeps , name_simulation):
    plt.figure()
    chi_sq = np.log10(chi_sq)
    plt.plot(sweeps , chi_sq)
    plt.axvline(x = number_equilibration_sweeps , color = 'blue')

    plt.xlabel("Eql sweeps: 1 to " + str(number_equilibration_sweeps) + " , Acc: " + str(number_equilibration_sweeps) + " to " + str(len(sweeps)))
    plt.ylabel("Log chi-Squared")
    plt.title("Trajectory Log(Chi-Squared) in Eq and Acc stages")
    plt.savefig("log_chi-squared_" + name_simulation , bbox_inches = 'tight')
    plt.show()


def plot_theta_sweeps(theta_matrix , sweeps , number_equilibration_sweeps , name_simulation , real_theta):
    plt.figure()
    plt.plot(sweeps , theta_matrix[:,0] , label = "theta_1")
    plt.plot(sweeps , theta_matrix[:,1] , label = "theta_2")
    plt.plot(sweeps , theta_matrix[:,2] , label = "theta_3")
    plt.axvline(x = number_equilibration_sweeps , color = 'blue')
    for theta in real_theta:
        plt.axhline(y = theta , color = 'orange')

    plt.xlabel("Eql sweeps: 1 to " + str(number_equilibration_sweeps) + " , Acc: " + str(number_equilibration_sweeps) + " to " + str(len(sweeps)))
    plt.ylabel("Thetas")
    plt.title("Trajectory Thetas in Eq and Acc stages")
    plt.legend()
    plt.savefig("thetas_" + name_simulation , bbox_inches = 'tight')
    plt.show()

def plot_hist_thetas(theta_matrix , name_simulation , max_theta , number_bins):
    plt.figure()
    plt.hist(theta_matrix[:,0] , range = (15 , 18) , histtype = 'barstacked' , rwidth = 0.85 , label = 'Theta_1')
    plt.xlabel("Theta 1 values")
    plt.ylabel("count (theta_1)")
    plt.title("Histogram: Theta 1 in Eql and Acc stages")
    plt.savefig("hist_theta_1_" + name_simulation , bbox_inches = 'tight')
    plt.show()

    plt.figure()
    plt.hist(theta_matrix[:, 1] , range = (50 , 52) , histtype = 'barstacked', rwidth = 0.85, label = 'Theta_2')
    plt.xlabel("Theta 2 values")
    plt.ylabel("count (theta_2)")
    plt.title("Histogram: Theta 2 in Eql and Acc stages")
    plt.savefig("hist_theta_2_" + name_simulation , bbox_inches = 'tight')
    plt.show()

    plt.figure()
    plt.hist(theta_matrix[:, 2] , range = (80 , 82) , histtype = 'barstacked', rwidth = 0.85,label = 'Theta_3')
    plt.xlabel("Theta 3 values")
    plt.ylabel("count (theta_3)")
    plt.title("Histogram: Theta 3 in Eql and Acc stages")
    plt.savefig("hist_theta_3_" + name_simulation , bbox_inches = 'tight')
    plt.show()


def plot_hist_chi_sq(chi_sq , name_simulation , max_theta , number_bins):
    plt.figure()
    plt.hist(chi_sq , range = (-10 , 10) , histtype = 'barstacked', rwidth = 0.85, label = 'Chi-squared')
    plt.xlabel("Chi-Squared values")
    plt.ylabel("count(chi-squared)")
    plt.title("Histogram: Chi-Squared in Eql and Acc stages")
    plt.savefig("hist_chi_sq_" + name_simulation , bbox_inches = 'tight')
    plt.show()


def plot_log_det_rank(file , name_simulation , number_points_plot):
    list_data_d_matrix = []
    list_data_e_matrix = []
    for index, line in enumerate(open(file)):
        if index != 0:
            data = line.split("|")
            list_data_d_matrix.append((data[0], float(data[1])))
            list_data_e_matrix.append((data[0], float(data[2].strip())))
    list_data_d_matrix = sorted(list_data_d_matrix, key=operator.itemgetter(1), reverse=True)
    list_data_e_matrix = sorted(list_data_e_matrix, key=operator.itemgetter(1), reverse=True)
    (_ , log_det_e) = zip(*list_data_e_matrix)
    (_ , log_det_d) = zip(*list_data_d_matrix)

    plt.figure()
    plt.plot(list(range(len(log_det_e))) , log_det_e)
    plt.xlabel("Rank")
    plt.ylabel("Log(det(E))")
    plt.title("Top " + str(number_points_plot) + " tuples correlation score")
    plt.savefig("rank_e_" + name_simulation, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(list(range(len(log_det_d))), log_det_d)
    plt.xlabel("Rank")
    plt.ylabel("Log(det(D))")
    plt.title("Top " + str(number_points_plot) + " tuples covariance score")
    plt.savefig("rank_d_" + name_simulation, bbox_inches='tight')
    plt.show()


def do_plots(name_simulation , number_replica , number_points_plot , number_equilibration_sweeps):

    #file_rank = name_simulation + "_labels_results"
    file_chi_sq = "data_plot_test_ham_" + str(number_replica) + "_" + name_simulation
    file_betas = "data_plot_test_beta_" + str(number_replica) + "_" + name_simulation
    #plot_log_det_rank(file_rank , name_simulation , number_points_plot)
    plot_chi_sq_sweeps(file_chi_sq , number_equilibration_sweeps , name_simulation)
    plot_betas_sweeps(file_betas , number_equilibration_sweeps , name_simulation)
