
# Circadian Model Analysis Program [CMAP] 
# Written by: Camillo Mizaikoff @ 2021
#
#  __main__ file: 
#  - Save/Load Simulation(s)
#  - Plot Data

#----------------------------------------------------------------------------------------------------------------------------------------------------

#TO DO:
# 
#
#

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Import Libraries

import sys
#Path to folder where model_analysis_module.py and model_config.py file is located
sys.path.append('/Users/camillo/Documents/Documents/Uni/Learning/Bachelor/Programming/modules/')

import time
#Start 'stopwatch' to monitor program runtime
start_time = time.clock()

import pickle
import numpy as np
import math as mt
import matplotlib.pyplot as plt 
import matplotlib.patches as pat 
import scipy.integrate as sp 
import scipy.optimize as op
import scipy.signal as si
import scipy.fftpack as fft
import scipy.interpolate as inter
import model_analysis_module_v9 as MAM
import almeida_model_config_standard as am

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Directory and folder path variables for saving/loading simulation files and saving figures

direct = '/Users/camillo/Documents/Documents/Uni/Learning/Bachelor/Models/Almeida/Standard Config/'
folder = 't-10000_params-standard_mod-10/'
model_version = ' - Standard Configuration (t = 10000)'

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Model Variables

#Length of simulation [h]
time_span = (0, 1000)
#Parameter perturbation factor
modulation = 0.1
#List of model differential variable names
names = am.var_names
#List of all variables
all_names = ['BMAL1', 'ROR', 'REV', 'DBP', 'E4BP4', 'CRY', 'PER', 'PER:CRY', 'CRY+', 'PER+']
#Time window to display in plots (Optional)
plot_time = (9902, 10000)

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Save/Load Simulation Data

#Sublime does not support user input so new simulation/load simulation is decided via changing of [newsim] variable
newsim = False
#Name of new simulation file or simulation file to be loaded
simname = 'almeida-standard_' + folder[:-1]

if newsim == True:
	
	#Get organized list of all model solutions for all parameter modulations [[mod_down], [mod_null], [mod_up]]
	modulated_solutions = MAM.Modulate_Parameters(am.almeida_circadian_clock, time_span, am.init_vals, am.param_var_name_list, 
						  						  am.param_val_list, am.param_names, modulation)

	#Save Modulate_Parameters object to file
	with open(direct + folder + simname + '.solutions', 'wb') as save_sim_file:
	 
		pickle.dump(modulated_solutions, save_sim_file)

elif newsim == False:

	#Load Modulated_Parameters object
	with open(direct + folder + simname + '.solutions', 'rb') as load_sim_file:
 
		modulated_solutions = pickle.load(load_sim_file)


#----------------------------------------------------------------------------------------------------------------------------------------------------
#Data Analysis

#Get per-cry delay data
per_cry_delay = MAM.Control_Analysis(modulated_solutions, 'Delay', names, 'PER+', 'CRY+').property_data
#Plot per-cry delay
MAM.Visualize_Data(direct, modulated_solutions, names).plot_delay_analysis(per_cry_delay, am.param_names, folder + 'Control Analysis/', 'PER+', 'CRY+')

for i in all_names:
	#Get amplitude + period length data for all differential variables
	amp_data = MAM.Control_Analysis(modulated_solutions, 'Amplitude', names, i).property_data
	per_data = MAM.Control_Analysis(modulated_solutions, 'Period', names, i).property_data
	#Plot parameter perturbations for all differential variables
	MAM.Visualize_Data(direct, modulated_solutions, names).plot_amp_per_analysis(amp_data, per_data, am.param_names, folder + 'Control Analysis/', i)

#Plot oscillations of all variables
MAM.Visualize_Data(direct, modulated_solutions, names).plot_diff_var(names, plot_time, folder + 'all_variables_', model_version = model_version)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_diff_var(names, plot_time, folder + 'all_variables_no_norm_', model_version = model_version, normalization = 'none')
#Plot oscillations of BMAL1, PER, CRY and PER:CRY
MAM.Visualize_Data(direct, modulated_solutions, names).plot_diff_var(['BMAL1', 'PER', 'CRY', 'PER:CRY'], plot_time, folder, model_version = model_version)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_diff_var(['BMAL1', 'PER', 'CRY', 'PER:CRY'], plot_time, folder + 'no_norm_', model_version = model_version, normalization = 'none')
#Plot oscillations of PER, CRY, PER:CRY, PER+ and CRY+
MAM.Visualize_Data(direct, modulated_solutions, names).plot_diff_var(['PER', 'CRY', 'PER:CRY', 'CRY+', 'PER+'], plot_time, folder + 'true-complex_', model_version = model_version)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_diff_var(['PER', 'CRY', 'PER:CRY', 'CRY+', 'PER+'], plot_time, folder + 'true-complex_no_norm_', model_version = model_version, normalization = 'none')

#Plot all variable oscillations for perturbed parameters
for i in names:
	for j in am.param_names:
		if i == 'PER:CRY':
			MAM.Visualize_Data(direct, modulated_solutions, names).plot_diff_var_mod(i, j, plot_time, folder + 'Variable Perturbations/' + 'PER-CRY/', model_version = model_version)
		else:
			MAM.Visualize_Data(direct, modulated_solutions, names).plot_diff_var_mod(i, j, plot_time, folder + 'Variable Perturbations/' + i + '/', model_version = model_version)

#Plot phase space(s)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_phase_space('BMAL1', 'PER', folder + 'Phase Space/', model_version = model_version)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_phase_space('BMAL1', 'CRY', folder + 'Phase Space/', model_version = model_version)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_phase_space('BMAL1', 'PER:CRY', folder + 'Phase Space/', model_version = model_version)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_phase_space('PER', 'CRY', folder + 'Phase Space/', model_version = model_version)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_phase_space('PER', 'PER:CRY', folder + 'Phase Space/', model_version = model_version)
MAM.Visualize_Data(direct, modulated_solutions, names).plot_phase_space('CRY', 'PER:CRY', folder + 'Phase Space/', model_version = model_version)

#Fourier Analysis
MAM.Visualize_Data(direct, modulated_solutions, names).plot_fourier('CRY')

#Period Doubling Debug
for j in names:
	MAM.Visualize_Data(direct, modulated_solutions, names).plot_peak_data(j, folder + 'Period Doubling/', model_version = model_version)

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Twist

#Get per-cry delay data
per_cry_delay = MAM.Control_Analysis(modulated_solutions, 'Delay', names, 'PER+', 'CRY+').property_data

delay_down = []
delay_up = []

cry_amp_del_down = []
cry_per_del_down = []
per_amp_del_down = []
per_per_del_down = []
cry_amp_del_up = []
cry_per_del_up = []
per_amp_del_up = []
per_per_del_up = []

for i in range(0, len(am.param_names)):
	if per_cry_delay[0][i] > per_cry_delay[1][i]:
		delay_down.append(1)
	else:
		delay_down.append(-1)

	if per_cry_delay[2][i] > per_cry_delay[1][i]:
		delay_up.append(1)
	else:
		delay_up.append(-1)

all_pers = []
all_amps = []

for i in all_names:
	#Get amplitude + period length data for all differential variables
	all_amps.append(MAM.Control_Analysis(modulated_solutions, 'Amplitude', names, i).property_data)
	all_pers.append(MAM.Control_Analysis(modulated_solutions, 'Period', names, i).property_data)

all_down_twists = []
all_up_twists = []

for j in range(0, len(all_names)):

	per_down = []
	per_up = []
	amp_down = []
	amp_up = []

	for i in range(0, len(am.param_names)):
		if all_pers[j][0][i] > all_pers[j][1][i]:
			per_down.append(1)
		else:
			per_down.append(-1)

		if all_pers[j][2][i] > all_pers[j][1][i]:
			per_up.append(1)
		else:
			per_up.append(-1)

		if all_amps[j][0][i] > all_amps[j][1][i]:
			amp_down.append(1)
		else:
			amp_down.append(-1)

		if all_amps[j][2][i] > all_amps[j][1][i]:
			amp_up.append(1)
		else:
			amp_up.append(-1)

	down_twist = []
	up_twist = []

	for i in range(0, len(am.param_names)):
		if per_down[i] == amp_down[i]:
			down_twist.append('+')
		else:
			down_twist.append('-')

		if per_up[i] == amp_up[i]:
			up_twist.append('+')
		else:
			up_twist.append('-')

		if all_names[j] == 'CRY+':
			if amp_down[i] == delay_down[i]:
				cry_amp_del_down.append('+')
			else:
				cry_amp_del_down.append('-')
			if amp_up[i] == delay_up[i]:
				cry_amp_del_up.append('+')
			else:
				cry_amp_del_up.append('-')

			if per_down[i] == delay_down[i]:
				cry_per_del_down.append('+')
			else:
				cry_per_del_down.append('-')
			if per_up[i] == delay_up[i]:
				cry_per_del_up.append('+')
			else:
				cry_per_del_up.append('-')

		if all_names[j] == 'PER+':
			if amp_down[i] == delay_down[i]:
				per_amp_del_down.append('+')
			else:
				per_amp_del_down.append('-')
			if amp_up[i] == delay_up[i]:
				per_amp_del_up.append('+')
			else:
				per_amp_del_up.append('-')

			if per_down[i] == delay_down[i]:
				per_per_del_down.append('+')
			else:
				per_per_del_down.append('-')
			if per_up[i] == delay_up[i]:
				per_per_del_up.append('+')
			else:
				per_per_del_up.append('-')

	all_down_twists.append(down_twist)
	all_up_twists.append(up_twist)

all_del_down_twists = [cry_amp_del_down, cry_per_del_down, per_amp_del_down, per_per_del_down]
all_del_up_twists = [cry_amp_del_up, cry_per_del_up, per_amp_del_up, per_per_del_up]
titles = ['PER+-CRY+ Delay - CRY+ Amplitude Twist', 'PER+-CRY+ Delay - CRY+ Period Twist', 'PER+-CRY+ Delay - PER+ Amplitude Twist', 'PER+-CRY+ Delay - PER+ Period Twist']

print('--------------------------------------------------------------------------------------------------------------------')
print('')

for i in range(0, len(all_names)):
	print('')
	print(all_names[i] + ' Amplitude-Period Twist')
	print('Param - 10%:', all_down_twists[i])
	print('Param + 10%:', all_up_twists[i])
	print('')

print('--------------------------------------------------------------------------------------------------------------------')

for i in range(0, 4):
	print('')
	print(titles[i])
	print('Param - 10%:', all_del_down_twists[i])
	print('Param + 10%:', all_del_up_twists[i])
	print('')

print('--------------------------------------------------------------------------------------------------------------------')

# for i in range(0, len(am.param_names)):
# 	print(am.param_names[i])
# 	for j in range(0, len(all_names)):
# 		print(all_down_twists[j][i])
# 		print(all_up_twists[j][i])

for i in range(0, len(am.param_names)):
	print(am.param_names[i])
	for j in range(0, 4):
		print(all_del_down_twists[j][i])
		print(all_del_up_twists[j][i])


#----------------------------------------------------------------------------------------------------------------------------------------------------
#Test Calculations

# window_low = MAM.Variable_Properties(modulated_solutions.organized_solutions[1][0], names, 'BMAL1', normalization = 'none').var_peak_indeces[-2]
# window_high = MAM.Variable_Properties(modulated_solutions.organized_solutions[1][0], names, 'BMAL1', normalization = 'none').var_peak_indeces[-1]

# BMAL = MAM.Variable_Properties(modulated_solutions.organized_solutions[1][0], names, 'BMAL1', normalization = 'none')
# CRY = MAM.Variable_Properties(modulated_solutions.organized_solutions[1][0], names, 'CRY', normalization = 'none')
# PERCRY = MAM.Variable_Properties(modulated_solutions.organized_solutions[1][0], names, 'PER:CRY', normalization = 'none')
# PER = MAM.Variable_Properties(modulated_solutions.organized_solutions[1][0], names, 'PER', normalization = 'none')


# BMAL_vals = np.array(BMAL.normed_solution[0][window_low:window_high])
# CRY_vals = np.array(CRY.normed_solution[5][window_low:window_high])
# PERCRY_vals = np.array(PERCRY.normed_solution[7][window_low:window_high])
# PER_vals = np.array(PERCRY.normed_solution[6][window_low:window_high])

# x_vals = np.linspace(0, 24, 298)

# all_names = ['BMAL1', 'ROR', 'REV', 'DBP', 'E4BP4', 'CRY', 'PER', 'PER:CRY', 'CRY+', 'PER+']
# for i in range(0, 10):
# 	var = MAM.Variable_Properties(modulated_solutions.organized_solutions[1][0], names, all_names[i], normalization = 'none')
# 	print(all_names[i] + ' Amplitude: ' + str(round(var.amplitude, 1)))

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Print program runtime
print (time.clock() - start_time, "seconds")




