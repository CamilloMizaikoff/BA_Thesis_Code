
# Module for Model Analysis [V7]
# Written by: Camillo Mizaikoff @ 2020

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Import Libraries

import sys
import numpy as np
import pickle
import math as mt
import scipy.integrate as sp 
import scipy.optimize as op
import scipy.signal as si
import scipy.fftpack as fft
import scipy.interpolate as inter
import matplotlib.pyplot as plt 
import matplotlib.patches as pat 

#----------------------------------------------------------------------------------------------------------------------------------------------------

def round_up_to(num, roundto):
	return(mt.ceil(num/roundto) * roundto)

class Variable_Properties:

	#--- Class Description ---	

	# Variable_Properties Class takes simulation data from a SciPy.solve_ivp() object and extracts/calculates properties of 
	# interest for a given model variable e.g. period length, amplitude, etc..


	#--- Class Argument Info ---

	# NECESSARY
	# solution [SciPy.solve_ivp() object]: Object that contains solution data of a model consisting of a system of differential equations.
	# name_list [List]: List of model's differential variable names.
	# var_name [String]: String dictating name of differential variable to be analyzed.

	# OPTIONAL
	# normalization [String]: String dictating normalization algorithm to be applied to data.
	#						  Can be 'mean', 'max' or 'none'.
	# stability [Tuple]: Tuple that defines min/max permitted deviation for peak stability check.
	# peak_prominence [Float]: Float that defines peak prominence used for peak finding algorithm.

	def __init__(self, solution, name_list, var_name, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):

		#Define class variables
		self.solution = solution
		self.name_list = name_list
		self.var_name = var_name
		self.normalization = normalization
		self.stability = stability
		self.peak_prominence = peak_prominence
		self.normed_solution = []
		self.simlen = self.solution.t[-1] - self.solution.t[0]

		#Dictionary with ODE variables and their indices to easily access in solution
		self.name_index_dict = {}
		for i in range(0, len(self.name_list)):
			self.name_index_dict[self.name_list[i]] = i
		if 'CRY' and 'PER:CRY' in self.name_list:
			self.name_index_dict['CRY+'] = len(self.name_index_dict)
		if 'PER' and 'PER:CRY' in self.name_list:
			self.name_index_dict['PER+'] = len(self.name_index_dict)
		#Index of differential variable being analyzed
		self.name_index = self.name_index_dict[self.var_name]

		#Apply normalization
		if self.normalization == 'mean':
			self.mean_normalize()
		elif self.normalization == 'max':
			self.max_normalize()
		elif self.normalization == 'none':
			self.no_normalize()

		#Add CRY + PER:CRY and PER + PER:CRY to solutions
		self.add_complex()

		#Given variable's peak indeces, heights and timestamps
		self.var_peak_indeces = self.get_peaks()
		self.var_peak_times = self.get_times()
		self.period_length = self.get_period()
		self.amplitude = self.get_amplitude()
		self.min_val = self.get_min()
		self.max_val = self.get_max()

		#Check oscillation stability
		self.is_stable()

		# Debugging to compare CRY + PER:CRY and PER + PER:CRY to CRY and PER timeseries
		# print(self.var_name, len(self.var_peak_times), len(self.var_peak_indeces))

		# self.points = [self.normed_solution[self.name_index_dict[self.var_name]][i] for i in self.var_peak_indeces]
		# print(len(self.var_peak_times), len(self.points))
		# print(self.var_peak_times, self.points)

		# plt.plot(self.solution.t, self.normed_solution[7], 'magenta', label = 'PER:CRY')
		# if self.var_name == 'PER+':
		# 	plt.plot(self.solution.t, self.normed_solution[6], 'cyan', label = 'PER')
		# 	plt.plot(self.solution.t, self.normed_solution[9], 'Salmon', label = 'PER+')
		# elif self.var_name == 'CRY+':
		# 	plt.plot(self.solution.t, self.normed_solution[5], 'blue', label = 'CRY')
		# 	plt.plot(self.solution.t, self.normed_solution[8], 'Red', label = 'CRY+')
		
		# plt.scatter(self.var_peak_times, self.points, color = 'green')
		# plt.xlim(900, 1000)
		# plt.legend()
		# plt.show()

	#Normalize model's expressions to their means
	def mean_normalize(self):
		for i in self.solution.y:
			self.y_vals = np.array(i)
			self.y_normed = self.y_vals/np.mean(self.y_vals)
			self.normed_solution.append(self.y_normed)

	#Normalize model's expressions to their max values
	def max_normalize(self):
		for i in self.solution.y:
			self.y_vals = np.array(i)
			self.y_normed = self.y_vals/np.max(self.y_vals)
			self.normed_solution.append(self.y_normed)

	#No Normalization of expressions
	def no_normalize(self):
		for i in self.solution.y:
			self.y_vals = np.array(i)
			self.normed_solution.append(self.y_vals)

	#Add CRY + PER:CRY and PER + PER:CRY to solutions
	def add_complex(self):
		if 'CRY' and 'PER:CRY' in self.name_list:
			self.total_cry = self.normed_solution[self.name_index_dict['CRY']] + self.normed_solution[self.name_index_dict['PER:CRY']]
			self.normed_solution.append(self.total_cry)

		if 'PER' and 'PER:CRY' in self.name_list:
			self.total_per = self.normed_solution[self.name_index_dict['PER']] + self.normed_solution[self.name_index_dict['PER:CRY']]
			self.normed_solution.append(self.total_per)


	#Find peaks in oscillation using SciPy find_peaks() function
	def get_peaks(self):
		self.peak_index = si.find_peaks(self.normed_solution[self.name_index], prominence = self.peak_prominence)
		return self.peak_index[0]

	#Get precise peak data using 3-point interpolation
	#Calculates maxima, for minima set find_min = True
	#Display interpolation points + curve by setting display_fit = True
	def interpolate_peak(self, peak_index, display_fit = False, find_min = False):
		#get 3 points before, on and after peak
		self.a = self.normed_solution[self.name_index][peak_index-1]
		self.a_t = self.solution.t[peak_index-1]
		self.b = self.normed_solution[self.name_index][peak_index]
		self.b_t = self.solution.t[peak_index]
		self.c = self.normed_solution[self.name_index][peak_index+1]
		self.c_t = self.solution.t[peak_index+1]

		self.x_vals = np.array([self.a_t, self.b_t, self.c_t])
		self.y_vals = np.array([self.a, self.b, self.c])

		#get factors of second order polynomial fit to points and calculate maximum of function
		self.quad_fit = np.polyfit(self.x_vals, self.y_vals, 2)
		self.poly = np.poly1d(self.quad_fit)
		self.find_max = op.minimize_scalar(-self.poly, bounds=(self.a_t, self.c_t), method='bounded')

		if find_min == True:
			self.find_max = op.minimize_scalar(self.poly, bounds=(self.a_t, self.c_t), method='bounded')

		self.peak_time = self.find_max.x
		self.peak_mag = self.poly(self.peak_time)
		self.peak_coords = (self.peak_time, self.peak_mag)
		
		if display_fit == True:
			#display points, fitted polynomial and calculated peak
			self.interp_times = np.linspace(self.a_t, self.c_t, 1000)
			plt.scatter(self.x_vals, self.y_vals, color = 'red')
			plt.plot(self.interp_times, [self.poly(i) for i in self.interp_times], color = 'blue')
			plt.scatter(abs(self.peak_time), abs(self.peak_mag), color = 'pink')
			plt.show()

		return self.peak_coords

	#Get approximate peak times by using peak indexes on solution.t
	def get_times(self):
		self.peak_times = []
		for i in self.var_peak_indeces:
			self.peak_times.append(self.solution.t[i])
		return self.peak_times
	
	#Get period of oscillation using peak-peak distance between last two peaks
	def get_period(self):
		self.last_peak = self.interpolate_peak(self.var_peak_indeces[-1])
		self.prev_peak = self.interpolate_peak(self.var_peak_indeces[-2])

		self.period = self.last_peak[0] - self.prev_peak[0]

		return self.period

	#Get period doubling period length using peak-peak distance between last two equal peaks
	def get_doubled_period(self):
		self.last_peak = self.interpolate_peak(self.var_peak_indeces[-1])
		self.sec_prev_peak = self.interpolate_peak(self.var_peak_indeces[-3])

		self.dub_period = self.last_peak[0] - self.sec_prev_peak[0]

		return self.dub_period

	#Get amplitude of normalized oscillation using last two peaks and amp = max - ((max+min)/2)
	def get_amplitude(self):
		self.peak_window_up = self.var_peak_indeces[-1] + 2
		self.peak_window_low = self.var_peak_indeces[-3] - 2
		self.window = self.normed_solution[self.name_index][self.peak_window_low:self.peak_window_up]
		self.window_list = self.window.tolist()
		self.max_window = max(self.window)
		self.min_window = min(self.window)
		self.max_index = self.peak_window_low + self.window_list.index(self.max_window)
		self.min_index = self.peak_window_low + self.window_list.index(self.min_window)

		self.high = self.interpolate_peak(self.max_index)[1]
		self.low = self.interpolate_peak(self.min_index, find_min = True)[1]
		self.amp = self.high - ((self.high + self.low)/2)

		# if self.var_name == 'PER' or self.var_name == 'CRY' and 'PER:CRY' in self.name_list:
		# 	self.complex = self.normed_solution[self.name_index_dict['PER:CRY']]
		# 	self.amp += self.complex[self.var_peak_indeces[-1]]

		return self.amp

	#Get minimum of normalized oscillation using last 2 periods
	def get_min(self):
		self.peak_window_up = self.var_peak_indeces[-1] + 2
		self.peak_window_low = self.var_peak_indeces[-3] - 1
		self.window = self.normed_solution[self.name_index][self.peak_window_low:self.peak_window_up]
		self.window_list = self.window.tolist()
		self.min_window = min(self.window)
		self.min_index = self.peak_window_low + self.window_list.index(self.min_window)
		self.mins = self.interpolate_peak(self.min_index, find_min = True)[1]
		return self.mins

	#Get Maximum of normalized oscillation using last 2 periods
	def get_max(self):
		self.peak_window_up = self.var_peak_indeces[-1] + 2
		self.peak_window_low = self.var_peak_indeces[-3] - 1
		self.window = self.normed_solution[self.name_index][self.peak_window_low:self.peak_window_up]
		self.window_list = self.window.tolist()
		self.max_window = max(self.window)
		self.max_index = self.peak_window_low + self.window_list.index(self.max_window)
		self.maxs = self.interpolate_peak(self.max_index)[1]
		return self.maxs

	#Check from which peak onward oscillation is within stability error margin
	#(Useful for finding period doubling or inconsistent oscillations. Requires some fine-tuning of stability arg...)
	def is_stable(self):
		self.var_peak_indeces = self.get_peaks()
		for i in range(len(self.var_peak_indeces)-1, 0, -1):
			self.error_margin = self.normed_solution[self.name_index][self.var_peak_indeces[i]]/self.normed_solution[self.name_index][self.var_peak_indeces[i-1]]
			if i == len(self.var_peak_indeces)-1 and (self.error_margin < self.stability[0] or self.error_margin > self.stability[1]):
				print('Oscillation of ' + self.var_name + ' does not meet stability parameters, check for peak splitting/chaos or choose longer simulation time!')
				return -1
				break
			elif self.error_margin >= self.stability[0] and self.error_margin <= self.stability[1]:
				continue
			else:
				self.stable_peak_time = self.solution.t[self.var_peak_indeces[i]]
				#print('First stable peak after ' + str(self.stable_peak_time) + ' hours.')
				return self.stable_peak_time
				break

	#Get list of [[all peak magnitudes][all period lengths]]
	def get_all_peak_data(self):

		self.magnitudes = []
		self.periods = []
		self.amplitudes = []

		self.global_min = self.get_min()

		for i in range(0, len(self.var_peak_indeces)):
			self.peak = self.interpolate_peak(self.var_peak_indeces[i])
			self.magnitudes.append(self.peak[1])
			self.amplitudes.append(self.peak[1] - ((self.peak[1] + self.global_min)/2))
			if i > 0:
				self.prev_peak = self.interpolate_peak(self.var_peak_indeces[i-1])
				self.periods.append(self.peak[0] - self.prev_peak[0])

		return [self.magnitudes, self.periods, self.amplitudes]

	#Debugging function to check if all peaks are accounted for when runnning calculations
	def lost_peaks(self): 
		self.prediction = self.simlen/self.period_length
		print('Prediction: ' + str(self.prediction))
		print('Peaks Found: ' + str(len(self.var_peak_indeces)))


#----------------------------------------------------------------------------------------------------------------------------------------------------

class Compare_Properties:

	#--- Class Description ---	

	# Compare_Properties Class takes simulation data from a SciPy.solve_ivp() object and compares properties of two variables
	# (currrently only phase delay)


	#--- Class Argument Info ---

	# NECESSARY
	# solution [SciPy.solve_ivp() object]: Object that contains solution data of a model consisting of a system of differential equations.
	# name_list [List]: List of model's differential variable names.
	# var_1 [String]: String dictating name of first differential variable to be compared.
	# var_2 [String]: String dictating name of second differential variable to be compared.

	# OPTIONAL
	# normalization [String]: String dictating normalization algorithm to be applied to data.
	#						  Can be 'mean', 'max' or 'none'.
	# stability [Tuple]: Tuple that defines min/max permitted deviation for peak stability check.
	# peak_prominence [Float]: Float that defines peak prominence used for peak finding algorithm.

	def __init__(self, solution, name_list, var_1, var_2, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):

		self.solution = solution
		self.name_list = name_list
		self.var_1 = var_1
		self.var_2 = var_2
		self.normalization = normalization
		self.stability = stability
		self.peak_prominence = peak_prominence

		#Get variable properties by initializing Variable_Properties class for each variable
		self.var_1_properties = Variable_Properties(self.solution, self.name_list, self.var_1, self.normalization, self.stability, self.peak_prominence)
		self.var_2_properties = Variable_Properties(self.solution, self.name_list, self.var_2, self.normalization, self.stability, self.peak_prominence)
		self.normed_solution = self.var_1_properties.normed_solution

		#Dictionary with ODE variables and their indices to easily access in solution
		self.name_index_dict = self.var_1_properties.name_index_dict
		self.var_1_index = self.name_index_dict[self.var_1]
		self.var_2_index = self.name_index_dict[self.var_2]


	#Get delay in oscillation of two variables from simulation
	def get_shifty(self):
		self.var_1_peak_times = self.var_1_properties.var_peak_times
		self.var_2_peak_times = self.var_2_properties.var_peak_times
		self.var_1_peak_indeces = self.var_1_properties.var_peak_indeces
		self.var_2_peak_indeces = self.var_2_properties.var_peak_indeces


		#Find which variable peaks first
		self.big_index = 0
		if self.var_1_peak_times[0] > self.var_2_peak_times[0]:
			self.big_index = len(self.var_1_peak_times) - 1
		else:
			self.big_index = len(self.var_2_peak_times) - 1

		self.var_1_peak_time = self.var_1_properties.interpolate_peak(self.var_1_peak_indeces[self.big_index])[0]
		self.var_2_peak_time = self.var_2_properties.interpolate_peak(self.var_2_peak_indeces[self.big_index])[0]
		self.peak_shift = abs(self.var_1_peak_time - self.var_2_peak_time)

		#self.peak_shifto = abs(self.var_1_peak_times[self.big_index]-self.var_2_peak_times[self.big_index])
		return self.peak_shift


#----------------------------------------------------------------------------------------------------------------------------------------------------

class Modulate_Parameters:

	#--- Class Description ---	

	# Modulate_Parameters Class takes a given ODE model and modulates each parameter up and down by the given modulation value, 
	# then sorts them in an array.


	#--- Class Argument Info ---

	# NECESSARY
	# model [Function]: Function that contains all equations of model in solve_ivp() compatible format
	# time_span [Tuple]: Tuple with (simulation start timestamp, simulation end timestamp)
	# init_vals [List]: List with initial values of all model variables
	# param_var_name_list [List]: List of all parameter variables
	# param_val_list [List]: List of all parameter variable values (in same order as param_var_name_list)
	# param_name_list [List]: List of all parameter variable names as STRINGS

	# OPTIONAL
	# modulation [Float]: Float value by which all parameters are modulated
	# no_mod [Bool]: Boolean operator that when set to 'True' disables parameter modulation and only returns standard solution

	def __init__(self, model, time_span, init_vals, param_var_name_list, param_val_list, param_name_list, modulation = 0.1, no_mod = False):

		self.model = model
		self.time_span = time_span
		self.init_vals = init_vals
		self.param_var_name_list = param_var_name_list
		self.param_val_list = param_val_list
		self.param_name_list = param_name_list
		self.modulation = modulation
		self.no_mod = no_mod

		#Dictionary of parameters and their indeces
		self.param_index_dict = {}
		for i in range(0, len(self.param_name_list)):
			self.param_index_dict[self.param_name_list[i]] = i

		#Creates list of all solutions for all parameter modulations
		self.all_solutions = self.get_solutions()
		#Creates organized solution list: [[mod_down][mod_null][mod_up]]
		self.organized_solutions = self.organize_solutions()


	#Solves model for each modulated parameter and returns list of solutions
	def get_solutions(self):
		self.all_param_solutions = []
		self.null_solution = sp.solve_ivp(self.model, self.time_span, self.init_vals)	

		if self.no_mod == True:
			return self.null_solution	

		for i in range(0, len(self.param_val_list)):

			self.param_mod_solutions = []

			self.param_val_list[i] = round(self.param_var_name_list[i] * (1-self.modulation), 3)
			self.solution = sp.solve_ivp(self.model, self.time_span, self.init_vals)
			self.param_mod_solutions.append(self.solution)

			self.param_mod_solutions.append(self.null_solution)

			self.param_val_list[i] = round(self.param_var_name_list[i] * (1+self.modulation), 3)
			self.solution = sp.solve_ivp(self.model, self.time_span, self.init_vals)
			self.param_mod_solutions.append(self.solution)

			
			self.param_val_list[i] = round(self.param_var_name_list[i] * 1, 3)
			self.all_param_solutions.append(self.param_mod_solutions)

			print('Parameter [' + str(i+1) + '/' + str(len(self.param_val_list)) + ']: ' + self.param_name_list[i] + ' - Complete')

		return self.all_param_solutions

	#Organize solution list in form of [[mod_down][no_mod][mod_up]]
	def organize_solutions(self):
		
		if self.no_mod == True:
			self.organized = [[], [self.all_solutions], []]
			return self.organized

		self.mod_down = []
		self.mod_null = []
		self.mod_up = []

		for param_solutions in self.all_solutions:
			self.mod_down.append(param_solutions[0])
			self.mod_null.append(param_solutions[1])
			self.mod_up.append(param_solutions[2])

		self.organized = [self.mod_down, self.mod_null, self.mod_up]
		return self.organized


#----------------------------------------------------------------------------------------------------------------------------------------------------

class Control_Analysis:

	#--- Class Description ---	

	# Control_Analysis Class takes a Modulate_Parameters Class object and analyzes effect of parameter modulation
	# on period lengths, amplitudes, delays, etc..


	#--- Class Argument Info ---

	# NECESSARY
	# modulate_parameters_object [Custom Object]: Object that contains organized solutions of model with various 
	#											  parameter modulations (see Modulate_Parameters Class)
	# model_property [String]: String dictating model property to by analyzed
	#						   Can be 'Period', 'Amplitude' or 'Delay'
	# name_list [List]: List of strings of model variable names
	# var_1 [String]: String dictating name of first differential variable to be compared.

	# OPTIONAL
	# var_2 [String]: String dictating name of second differential variable to be compared when 'Delay' is selected.
	# normalization [String]: String dictating normalization algorithm to be applied to data.
	#						  Can be 'mean', 'max' or 'none'.
	# stability [Tuple]: Tuple that defines min/max permitted deviation for peak stability check.
	# peak_prominence [Float]: Float that defines peak prominence used for peak finding algorithm.

	def __init__(self, modulate_parameters_object, model_property, name_list, var_1, var_2 = '', normalization = 'mean', 
				 stability = (0.99, 1.01), peak_prominence = 0.5):

		self.modulate_parameters_object = modulate_parameters_object
		self.organized_solutions = self.modulate_parameters_object.organized_solutions
		self.model_property = model_property
		self.name_list = name_list
		self.var_1 = var_1
		self.var_2 = var_2
		self.normalization = normalization
		self.stability = stability
		self.peak_prominence = peak_prominence

		#Check to make sure two variables are given for Delay analysis
		if self.model_property == 'Delay' and self.var_2 == '':
			sys.exit('Error: Please specify second variable for delay comparison!')

		#Dictionary with ODE variables and their indices to easily access in solution
		self.name_index_dict = {}
		for i in range(0, len(self.name_list)):
			self.name_index_dict[self.name_list[i]] = i
		self.name_index_dict['CRY+'] = len(self.name_index_dict)
		self.name_index_dict['PER+'] = len(self.name_index_dict)
		self.var_1_index = self.name_index_dict[self.var_1]
		if self.var_2 in self.name_index_dict:
			self.var_2_index = self.name_index_dict[self.var_2]

		#Dictionary of possible model properties to analyze and corresponding functions
		self.model_property_dict = {
			'Period' : self.analyze_period, 
			'Amplitude' : self.analyze_amplitude,
			'Delay' : self.analyze_delay,
			#'Stable Time' : self.analyze_stability,
		}

		#Run selected function
		self.property_data = self.get_property()


	#Finds property to by analyzed in dictionary and runs corresponding function
	def get_property(self):
		if self.model_property in self.model_property_dict:
			self.daterino = self.model_property_dict[self.model_property]()
			return self.daterino

	#Gets period length of var_1 for all parameter modulations and returns list of arrays with values according to organized solution list
	def analyze_period(self):
		self.down = []
		self.null = []
		self.up = []
		self.iterate_list = [self.down, self.null, self.up]

		for i in range(0, 3):
			if i == 1: 
				self.seed = self.organized_solutions[i][0]
				self.period_len = Variable_Properties(self.seed, self.name_list, self.var_1, self.normalization, self.stability, self.peak_prominence).period_length
				for j in range(0, len(self.iterate_list[0])):
					self.iterate_list[i].append(self.period_len)
			else:
				for sol in self.organized_solutions[i]:
					self.period_len = Variable_Properties(sol, self.name_list, self.var_1, self.normalization, self.stability, self.peak_prominence).period_length
					self.iterate_list[i].append(self.period_len)

		self.period_array_list = [np.array(self.iterate_list[0]), np.array(self.iterate_list[1]), np.array(self.iterate_list[2])]
		return self.period_array_list

	#Gets amplitude of var_1 for all parameter modulations and returns list of arrays with values according to organized solution list
	def analyze_amplitude(self):
		self.down = []
		self.null = []
		self.up = []
		self.iterate_list = [self.down, self.null, self.up]

		for i in range(0, 3):
			if i == 1: 
				self.seed = self.organized_solutions[i][0]
				self.rel_amp = Variable_Properties(self.seed, self.name_list, self.var_1, self.normalization, self.stability, self.peak_prominence).amplitude
				for j in range(0, len(self.iterate_list[0])):
					self.iterate_list[i].append(self.rel_amp)
			else:
				for sol in self.organized_solutions[i]:
					self.rel_amp = Variable_Properties(sol, self.name_list, self.var_1, self.normalization, self.stability, self.peak_prominence).amplitude
					self.iterate_list[i].append(self.rel_amp)

		self.amp_array_list = [np.array(self.iterate_list[0]), np.array(self.iterate_list[1]), np.array(self.iterate_list[2])]
		return self.amp_array_list

	#Gets delay of var_1 and var_2 for all parameter modulations and returns list of arrays with values according to organized solution list
	def analyze_delay(self):
		self.down = []
		self.null = []
		self.up = []
		self.iterate_list = [self.down, self.null, self.up]

		for i in range(0, 3):
			if i == 1: 
				self.seed = self.organized_solutions[i][0]
				self.shiftyshit = Compare_Properties(self.seed, self.name_list, self.var_1, self.var_2, self.normalization, self.stability, self.peak_prominence).get_shifty()
				for j in range(0, len(self.iterate_list[0])):
					self.iterate_list[i].append(self.shiftyshit)
			else:
				for sol in self.organized_solutions[i]:
					self.shiftyshit = Compare_Properties(sol, self.name_list, self.var_1, self.var_2, self.normalization, self.stability, self.peak_prominence).get_shifty()
					self.iterate_list[i].append(self.shiftyshit)

		self.shift_array_list = [np.array(self.iterate_list[0]), np.array(self.iterate_list[1]), np.array(self.iterate_list[2])]
		return self.shift_array_list


#----------------------------------------------------------------------------------------------------------------------------------------------------

class Factor_Analysis:

	#--- Class Description ---	

	# Factor_Analysis Class iterates over modulate_parameters_objects in a directory and extracts data to analyze
	# effect(s) of inhibition factor changes on model properties

	#--- Class Argument Info ---

	# NECESSARY
	# directory [String]: String defining directory in which model_property_objects are saved
	# folder [String]: String of folder name files are located in
	# file_pre [String]: String of file prefix for modulate_parameters_object save file names
	# factor_range [List]: List of inhibition factor values

	# OPTIONAL
	# sorter [String]: String defining if files are sorted...
	#				   'loose' (all in the same folder) or
	#				   'folder' (in individual folders)
	# (I can't quite remember why I implemented this, best to put all files in same folder and keep set to 'loose'...)

	def __init__(self, directory, folder, file_pre, factor_range, sorter = 'loose'):

		self.directory = directory
		self.folder = folder
		self.file_pre = file_pre
		self.factor_range = factor_range
		self.factor_count = len(self.factor_range)
		self.sorter = sorter
		self.mod_param_obj_list = self.get_solutions()
		self.org_sol_list = self.get_organized()

	#Iterates over all modulate_paramters_object files and appends them to mod_obj list
	def get_solutions(self):
		self.mod_obj = []
		for i in self.factor_range:

			self.counter = str(i)
			if type(i) == float:
				self.counter = str(i).replace('.', ',')

			if self.sorter == 'folder':
				self.path = self.directory + self.folder + self.counter + '/' + self.file_pre + self.folder + self.counter + '.solutions'
			elif self.sorter == 'loose':
				self.path = self.directory + self.file_pre + self.folder + self.counter + '.solutions'

			#Load Modulated_Parameters object
			with open(self.path, 'rb') as self.load_sim_file:
	 
				self.modulated_solutions = pickle.load(self.load_sim_file)
				self.mod_obj.append(self.modulated_solutions)

		return self.mod_obj

	#Iterates over mod_obj list and extracts all organized solutions lists, then appends them to org_sol list
	def get_organized(self):
		self.org_sol = []
		for i in self.mod_param_obj_list:
			self.org_sol.append(i.organized_solutions)

		return self.org_sol

	#Iterates over all solutions in org_sol list and extracts delay between var_1 and var_2 (see Compare_Properties Class)
	#For solutions with modulated parameters, 'mod' can be set to: ['Param Name', '-' or '+' for up/down modulation respectively]
	def get_delays(self, name_list, var_1, var_2, mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.delays = []
		for i in range(0, self.factor_count):
			self.modulate_parameters_object = self.mod_param_obj_list[i]
			self.organized_solutions = self.org_sol_list[i]
			self.param_index = 0
			self.mod_index = 1
			# self.var_1_index = self.name_index_dict[var_1]
			# self.var_2_index = self.name_index_dict[var_2]
			
			if mod != None:
				self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
				if mod[1] == '-':
					self.mod_index = 0
				elif mod[1] == '+':
					self.mod_index = 2

			self.solution = self.organized_solutions[self.mod_index][self.param_index]
			self.shift = Compare_Properties(self.solution, name_list, var_1, var_2, normalization, stability, peak_prominence).get_shifty()
			self.delays.append(self.shift)

		return self.delays

	#Iterates over all solutions in org_sol list and extracts period lengths of var_1 (see Variable_Properties Class)
	#For solutions with modulated parameters, 'mod' can be set to: ['Param Name', '-' or '+' for up/down modulation respectively]
	def get_periods(self, name_list, var_1, mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.periods = []
		for i in range(0, self.factor_count):
			self.modulate_parameters_object = self.mod_param_obj_list[i]
			self.organized_solutions = self.org_sol_list[i]
			self.param_index = 0
			self.mod_index = 1
			# self.var_1_index = self.name_index_dict[var_1]
			# self.var_2_index = self.name_index_dict[var_2]
			
			if mod != None:
				self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
				if mod[1] == '-':
					self.mod_index = 0
				elif mod[1] == '+':
					self.mod_index = 2

			self.solution = self.organized_solutions[self.mod_index][self.param_index]
			self.period = Variable_Properties(self.solution, name_list, var_1, normalization, stability, peak_prominence).period_length
			self.periods.append(self.period)

		return self.periods

	#Iterates over all solutions in org_sol list and extracts period lengths of var_1 with period doubling (see Variable_Properties Class)
	#For solutions with modulated parameters, 'mod' can be set to: ['Param Name', '-' or '+' for up/down modulation respectively]
	def get_dub_periods(self, name_list, var_1, mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.dub_periods = []
		for i in range(0, self.factor_count):
			self.modulate_parameters_object = self.mod_param_obj_list[i]
			self.organized_solutions = self.org_sol_list[i]
			self.param_index = 0
			self.mod_index = 1
			# self.var_1_index = self.name_index_dict[var_1]
			# self.var_2_index = self.name_index_dict[var_2]
			
			if mod != None:
				self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
				if mod[1] == '-':
					self.mod_index = 0
				elif mod[1] == '+':
					self.mod_index = 2

			self.solution = self.organized_solutions[self.mod_index][self.param_index]
			self.dub_period = Variable_Properties(self.solution, name_list, var_1, normalization, stability, peak_prominence).get_doubled_period()
			self.dub_periods.append(self.dub_period)

		return self.dub_periods

	#Iterates over all solutions in org_sol list and extracts magnitudes of var_1 (see Variable_Properties Class)
	#For solutions with modulated parameters, 'mod' can be set to: ['Param Name', '-' or '+' for up/down modulation respectively]
	def get_all_peak_mags(self, name_list, var_1, amp = False, mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.all_mags = []
		for i in range(0, self.factor_count):
			self.modulate_parameters_object = self.mod_param_obj_list[i]
			self.organized_solutions = self.org_sol_list[i]
			self.param_index = 0
			self.mod_index = 1
			# self.var_1_index = self.name_index_dict[var_1]
			# self.var_2_index = self.name_index_dict[var_2]
			
			if mod != None:
				self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
				if mod[1] == '-':
					self.mod_index = 0
				elif mod[1] == '+':
					self.mod_index = 2

			self.solution = self.organized_solutions[self.mod_index][self.param_index]
			self.peak_mags = Variable_Properties(self.solution, name_list, var_1, normalization, stability, peak_prominence).get_all_peak_data()
			if amp == False:
				self.all_mags.append(self.peak_mags[0])
			if amp == True:
				self.all_mags.append(self.peak_mags[2])

		return self.all_mags


#----------------------------------------------------------------------------------------------------------------------------------------------------

class Mod_Analysis:

	#--- Class Description ---	

	# Mod_Analysis Class iterates over modulate_parameters_objects in a directory and extracts data to analyze
	# effect(s) of graduated single parameter changes on model properties

	#--- Class Argument Info ---

	# NECESSARY
	# directory [String]: String defining directory in which model_property_objects are saved
	# file_pre [String]: String of file prefix for modulate_parameters_object save file names
	# mod_range [List]: List of discrete parameter modulation values 

	# (Pretty much just an appropriated clone of the Factor_Analysis Class)

	def __init__(self, directory, file_pre, mod_range):

		self.directory = directory
		self.file_pre = file_pre
		self.mod_range = mod_range
		self.mod_count = len(self.mod_range)
		self.mod_param_obj_list = self.get_solutions()
		self.org_sol_list = self.get_organized()

	#Iterates over all modulate_paramters_object files and appends them to mod_obj list
	def get_solutions(self):
		self.mod_obj = []
		for i in self.mod_range:

			self.counter = str(i)
			if type(i) == float:
				self.counter = str(i).replace('.', ',')

			# if self.sorter == 'folder':
			# 	self.path = self.directory + self.folder + self.counter + '/' + self.file_pre + self.folder + self.counter + '.solutions'
			# elif self.sorter == 'loose':
			self.path = self.directory + self.file_pre + self.counter + '.solutions'

			#Load Modulated_Parameters object
			with open(self.path, 'rb') as self.load_sim_file:
	 
				self.modulated_solutions = pickle.load(self.load_sim_file)
				self.mod_obj.append(self.modulated_solutions)

		return self.mod_obj

	#Iterates over mod_obj list and extracts all organized solutions lists, then appends them to org_sol list
	def get_organized(self):
		self.org_sol = []
		for i in self.mod_param_obj_list:
			self.org_sol.append(i.organized_solutions)

		return self.org_sol

	#Iterates over all solutions in org_sol list and extracts delay between var_1 and var_2 (see Compare_Properties Class)
	def get_delays(self, name_list, var_1, var_2, mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.delays = []
		for i in range(0, self.mod_count):
			self.modulate_parameters_object = self.mod_param_obj_list[i]
			self.organized_solutions = self.org_sol_list[i]
			self.param_index = 0
			self.mod_index = 1
			# self.var_1_index = self.name_index_dict[var_1]
			# self.var_2_index = self.name_index_dict[var_2]
			
			# if mod != None:
			# 	self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
			# 	if mod[1] == '-':
			# 		self.mod_index = 0
			# 	elif mod[1] == '+':
			# 		self.mod_index = 2

			self.solution = self.organized_solutions[self.mod_index][self.param_index]
			self.shift = Compare_Properties(self.solution, name_list, var_1, var_2, normalization, stability, peak_prominence).get_shifty()
			self.delays.append(self.shift)

		return self.delays

	#Iterates over all solutions in org_sol list and extracts periods of var_1 (see Variable_Properties Class)
	def get_periods(self, name_list, var_1, mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.periods = []
		for i in range(0, self.mod_count):
			self.modulate_parameters_object = self.mod_param_obj_list[i]
			self.organized_solutions = self.org_sol_list[i]
			self.param_index = 0
			self.mod_index = 1
			# self.var_1_index = self.name_index_dict[var_1]
			# self.var_2_index = self.name_index_dict[var_2]
			
			# if mod != None:
			# 	self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
			# 	if mod[1] == '-':
			# 		self.mod_index = 0
			# 	elif mod[1] == '+':
			# 		self.mod_index = 2

			self.solution = self.organized_solutions[self.mod_index][self.param_index]
			self.period = Variable_Properties(self.solution, name_list, var_1, normalization, stability, peak_prominence).period_length
			self.periods.append(self.period)

		return self.periods

	#Iterates over all solutions in org_sol list and extracts amplitudes of var_1 (see Variable_Properties Class)
	def get_amplitudes(self, name_list, var_1, mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.amplitudes = []
		for i in range(0, self.mod_count):
			self.modulate_parameters_object = self.mod_param_obj_list[i]
			self.organized_solutions = self.org_sol_list[i]
			self.param_index = 0
			self.mod_index = 1
			# self.var_1_index = self.name_index_dict[var_1]
			# self.var_2_index = self.name_index_dict[var_2]
			
			# if mod != None:
			# 	self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
			# 	if mod[1] == '-':
			# 		self.mod_index = 0
			# 	elif mod[1] == '+':
			# 		self.mod_index = 2

			self.solution = self.organized_solutions[self.mod_index][self.param_index]
			self.amplitude = Variable_Properties(self.solution, name_list, var_1, normalization, stability, peak_prominence).amplitude
			self.amplitudes.append(self.amplitude)

		return self.amplitudes

#----------------------------------------------------------------------------------------------------------------------------------------------------

class Visualize_Data:

	#--- Class Description ---	

	# Visualize_Data Class generates plots and graphs to visualize the various kinds of data gathered using 
	# the other classes and functions in this module

	#--- Class Argument Info ---

	# NECESSARY
	# directory [String]: String defining directory in which model_property_objects are saved
	# modulate_parameters_object [Custom Object]: Object that contains organized solutions of model with various 
	#											  parameter modulations (see Modulate_Parameters Class)
	# name_list [List]: List of strings of all model variable names

	# OPTIONAL
	# buffalo [Int]: Int for buffer to max/min values on Y axis
	# (I don't remember why I implemented this, I dont think it ever actually gets used...)

	def __init__(self, directory, modulate_parameters_object, name_list, buffalo = 10):

		self.directory = directory
		self.modulate_parameters_object = modulate_parameters_object
		self.organized_solutions = self.modulate_parameters_object.organized_solutions
		self.modulation = self.modulate_parameters_object.modulation
		self.name_list = name_list
		self.buffalo = buffalo

		#Lots and lots and lots of pretty colors <3
		self.colors = ['LightSalmon', 'Coral', 'Tomato', 'OrangeRed', 'Red', 'Crimson', 'FireBrick', 'Orange', 'MediumOrchid', 'DarkViolet', 'DarkMagenta', 'Indigo', 'MediumBlue', 'RoyalBlue', 'DodgerBlue', 'CadetBlue', 'Turquoise', 'Teal', 'DarkOliveGreen', 'Olive', 'Green', 'DarkGreen', 'SpringGreen', 'GreenYellow']
		self.sub_colors = ['Red', 'Sienna', 'Orange', 'olive', 'green', 'teal', 'magenta', 'purple', 'indigo']
		self.quad_colors = ['Red', 'Blue', 'Green', 'magenta']
		self.tri_colore = ['tomato', 'olive', 'cadetblue']
		self.strong_tri_colore = ['red', 'green', 'blue']
		self.more_colors = ['steelblue', 'olive', 'tomato']

		self.scatter_markerinos = ['v', 'o', '^']
		self.control_labelinos = ['Parameter - ', 'Baseline', 'Parameter + ']
		self.dev_boxes = [0.01, 0.03, 0.05, 0.1]

		#Dictionary with ODE variables and their indices to easily access in solution
		self.name_index_dict = {}
		for i in range(0, len(self.name_list)):
			self.name_index_dict[self.name_list[i]] = i
		self.name_index_dict['CRY+'] = len(self.name_index_dict)
		self.name_index_dict['PER+'] = len(self.name_index_dict)

	#Get min and max values of prop_data arg
	def get_min_max(self, prop_data):
		self.maxes = []
		self.mins = []
		for i in prop_data:
			self.maxes.append(max(i))
			self.mins.append(min(i))
		self.min_max_list = [min(self.mins), max(self.maxes)]
		return self.min_max_list

	#Get Y-axis lims for prop_data 
	def get_lims(self, prop_data):
		self.base = prop_data[1][0]
		self.bounds = self.get_min_max(prop_data)
		self.ranger = self.bounds[1] - self.bounds[0]
		self.buff = self.ranger/self.buffalo

		self.min_diff = abs(self.base-self.bounds[0])
		self.max_diff = abs(self.bounds[1]-self.base)
		self.diffs = [self.min_diff, self.max_diff]
		self.newmax = max(self.diffs)
		self.newlims = [self.base - self.newmax - self.buff, self.base + self.newmax + self.buff]
		return self.newlims

	#Plot delay control analysis (prop_data = Control_Analysis('delay').property_data object)
	def plot_delay_analysis(self, prop_data, param_names, folder, var_1, var_2, box_lim = 4):
		self.lims = self.get_lims(prop_data)
		self.base = prop_data[1][0]

		self.fig = plt.figure(figsize = (15,10))
		self.fig.suptitle('Almeida 8 ODE model - Control Analysis [' + var_1 + ' - ' + var_2 + ']', fontsize = 20)

		self.ax1 = self.fig.add_subplot(1, 1, 1)
		self.ax1.set_title('Delay (Baseline = ' + str(round(prop_data[1][0], 1)) + 'h)', fontsize = 19, pad = 10)

		for i in range(2, -1, -2):
			self.ax1.scatter(param_names, prop_data[i], s = 30, c = self.strong_tri_colore[i], marker = self.scatter_markerinos[i], label = self.control_labelinos[i] + str(self.modulation * 100) + '%')
			self.ax1.plot(param_names, prop_data[i], c = self.tri_colore[i], linestyle = 'solid', alpha = 0.5, linewidth = 1)
		self.ax1.plot(param_names, prop_data[1], c = self.tri_colore[1], linestyle = 'solid', alpha = 1, linewidth = 1, label = self.control_labelinos[1])

		self.ax1.legend(loc = 'upper right',fontsize = 15)
		self.ax1.set_ylabel('Peak Delay [h]', fontsize = 18)
		self.ax1.set_xlabel('Model Parameters', fontsize = 18, labelpad = 15)
		self.ax1.tick_params('both', which = 'major', labelsize = 13)
		#self.ax1.tick_params('both', which = 'minor', labelsize = 10)
		#self.ax1.set_yticks(np.linspace(0, 5, 11), minor = 'true')
		self.ax1.set_ylim(self.lims)
		self.ax1.grid(which = 'major', axis = 'x', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')
		for i in range(0, box_lim):
			self.ax1.add_patch(pat.Rectangle((0, self.base*(1 - self.dev_boxes[i])), len(param_names)-1, 
											 (self.base * (1 + self.dev_boxes[i]) - self.base * (1 - self.dev_boxes[i])), 
											  alpha = 0.075, color = 'green'))

		plt.savefig(self.directory + folder + 'control_analysis_' + var_1 + ' - ' + var_2 + '_delay' + '.png')
		plt.close(self.fig)

	#Plot Amplitude & Period control analysis (prop_data = Control_Analysis('amplitude'/'period').property_data object)
	def plot_amp_per_analysis(self, amp_prop_data, per_prop_data, param_names, folder, var_1, box_lim = 4):
		self.amp_lims = self.get_lims(amp_prop_data)
		self.per_lims = self.get_lims(per_prop_data)
		self.amp_base = amp_prop_data[1][0]
		self.per_base = per_prop_data[1][0]

		#Plot Variable Amplitude/Period Control Analysis
		self.fig = plt.figure(figsize = (15,10))
		self.fig.suptitle('Almeida 8 ODE model - Control Analysis [' + var_1 + ']', fontsize = 20)

		self.ax1 = self.fig.add_subplot(2, 1, 1)
		self.ax1.set_title('Amplitude (Baseline = ' + str(round(amp_prop_data[1][0], 2)) + 'a.u.)', fontsize = 19)

		#Plot amplitude data
		for i in range(2, -1, -2):
			self.ax1.scatter(param_names, amp_prop_data[i], s = 30, c = self.strong_tri_colore[i], marker = self.scatter_markerinos[i], label = self.control_labelinos[i] + str(self.modulation * 100) + '%')
			self.ax1.plot(param_names, amp_prop_data[i], c = self.tri_colore[i], linestyle = 'solid', alpha = 0.5, linewidth = 1)
		self.ax1.plot(param_names, amp_prop_data[1], c = self.tri_colore[1], linestyle = 'solid', alpha = 1, linewidth = 1, label = self.control_labelinos[1])

		self.ax1.legend(loc = 'upper right',fontsize = 15)
		self.ax1.set_ylabel('Amplitude [a.u.]', fontsize = 18)
		self.ax1.tick_params('both', which = 'major', labelsize = 13)
		#self.ax1.tick_params('both', which = 'minor', labelsize = 10)
		#self.ax1.set_yticks(np.linspace(0, 5, 11), minor = 'true')
		self.ax1.set_ylim(self.amp_lims)
		plt.setp(self.ax1.get_xticklabels(), visible=False)
		self.ax1.grid(which = 'major', axis = 'x', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')
		for i in range(0, box_lim):
			self.ax1.add_patch(pat.Rectangle((0, self.amp_base*(1 - self.dev_boxes[i])), len(param_names)-1, 
											 (self.amp_base * (1 + self.dev_boxes[i]) - self.amp_base * (1 - self.dev_boxes[i])), 
											  alpha = 0.075, color = 'green'))

		self.ax2 = self.fig.add_subplot(2, 1, 2)
		self.ax2.set_title('Period (Baseline = ' + str(round(per_prop_data[1][0], 1)) + 'h)', fontsize = 19)

		#Plot Period Data
		for i in range(2, -1, -2):
			self.ax2.scatter(param_names, per_prop_data[i], s = 30, c = self.strong_tri_colore[i], marker = self.scatter_markerinos[i], label = self.control_labelinos[i] + str(self.modulation * 100) + '%')
			self.ax2.plot(param_names, per_prop_data[i], c = self.tri_colore[i], linestyle = 'solid', alpha = 0.5, linewidth = 1)
		self.ax2.plot(param_names, per_prop_data[1], c = self.tri_colore[1], linestyle = 'solid', alpha = 1, linewidth = 1, label = self.control_labelinos[1])

		self.ax2.set_ylabel('Period Length [h]', fontsize = 18)
		self.ax2.set_xlabel('Model Parameters', fontsize = 18, labelpad = 15)
		self.ax2.tick_params('both', which = 'major', labelsize = 13)
		#self.ax2.tick_params('both', which = 'minor', labelsize = 10)
		#self.ax2.set_yticks(np.linspace(0, 5, 11), minor = 'true')
		self.ax2.set_ylim(self.per_lims)
		self.ax2.grid(which = 'major', axis = 'x', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax2.set_facecolor('whitesmoke')
		for i in range(0, box_lim):
			self.ax2.add_patch(pat.Rectangle((0, self.per_base*(1 - self.dev_boxes[i])), len(param_names)-1, 
											 (self.per_base * (1 + self.dev_boxes[i]) - self.per_base * (1 - self.dev_boxes[i])), 
											  alpha = 0.075, color = 'green'))

		plt.savefig(self.directory + folder + 'control_analysis_' + var_1 + '_amp-per_no-norm' + '.png')
		plt.close(self.fig)

	#Plot time series of variables in var_list
	def plot_diff_var(self, var_list, time_range, folder, model_version = '', normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.solution = self.organized_solutions[1][0]
		self.max_ex = []
		self.y_tit = ''
		if normalization == 'mean':
			self.y_tit = 'Mean-Normalized'
		elif normalization == 'max':
			self.y_tit = 'Max-Normalized'
		elif normalization == 'none':
			self.y_tit = ''

		self.fig = plt.figure(figsize = (15,10))

		#self.fig.suptitle('Almeida 8 ODE Model' + model_version, fontsize = 20)
		self.ax1 = self.fig.add_subplot(1, 1, 1)
		#self.ax1.set_title(r'Competetive Inhibtion ($\alpha$ = 70)', fontsize = 19, pad = 20)

		for i in range(0, len(var_list)):

			self.var_name = var_list[i]
			self.var_props = Variable_Properties(self.solution, self.name_list, self.var_name, normalization, stability, peak_prominence)
			self.ax1.plot(self.solution.t, self.var_props.normed_solution[self.name_index_dict[self.var_name]], c = self.sub_colors[i], linestyle = 'solid', linewidth = 1.5, label = self.var_name)
			self.max_ex.append(self.var_props.max_val)
			self.max_ex.append(self.var_props.min_val)

		self.ax1.legend(loc = 'upper right',fontsize = 15)
		self.ax1.set_ylabel(self.y_tit + ' Expression [a.u.]', fontsize = 20)
		self.ax1.set_xlabel('Time [h]', fontsize = 20)
		self.ax1.tick_params('both', which = 'major', labelsize = 20)
		self.ax1.tick_params('both', which = 'minor', labelsize = 20)
		# self.ax1.set_yticks(np.linspace(0, max(self.max_ex), max(self.max_ex)+1), minor = 'true')
		# self.ax1.set_yticks(np.linspace(0, 10, 11), minor = 'true')
		# self.ax1.set_xticks(np.linspace(0, 1000*24, (1000*24/6)+1), minor = 'true')
		self.ax1.set_xticks([i for i in range(time_range[0], time_range[1] + 1, 6)], minor = 'true')
		self.ax1.set_xticks([i for i in range(time_range[0], time_range[1], 24)])

		self.lim_up = max(self.max_ex)+(max(self.max_ex)/10)
		self.lim_down = min(self.max_ex) - (min(self.max_ex)/10)
		if self.lim_down < 0:
			self.lim_down = 0

		self.ax1.set_ylim([self.lim_down, self.lim_up])
		self.ax1.set_xlim(time_range)
		#self.ax1.grid(which = 'major', color = 'grey', linestyle = '-', linewidth = 0.75)
		#self.ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')

		self.label_test = []
		for i in range(0, 9877, 24):
			self.label_test.append(i)
		for i in range(0, 121, 24):
			self.label_test.append(i)

		self.ax1.set_xticklabels(self.label_test)

		#plt.tight_layout()
		#plt.show()
		plt.savefig(self.directory + folder + 'model_oscillation.png')
		plt.close(self.fig)
		
	#Plot time series of variable 'var_name' for parameter 'param_name' modulations
	def plot_diff_var_mod(self, var_name, param_name, time_range, folder, model_version = '', hide = [], normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		self.param_index = self.modulate_parameters_object.param_index_dict[param_name]
		self.solution_list = self.modulate_parameters_object.all_solutions[self.param_index]
		self.max_ex = []
		self.lab = [param_name + ' - ' + str(self.modulation*100) + '%',
					param_name + ' = ' + str(self.modulate_parameters_object.param_val_list[self.param_index]),  
					param_name + ' +' + str(self.modulation*100) + '%']
		self.y_tit = ''
		if normalization == 'mean':
			self.y_tit = 'Mean-Normalized'
		elif normalization == 'max':
			self.y_tit = 'Max-Normalized'
		elif normalization == 'none':
			self.y_tit = ''

		#Hide series from plot
		self.range_index = (0, 3, 1)
		if len(hide) == 1:
			if '-' in hide:
				self.range_index = (1, 3, 1)
			elif '=' in hide:
				self.range_index = (0, 3, 2)
			elif '+' in hide:
				self.range_index = (0, 2, 1)
		elif len(hide) == 2:
			if '-' in hide and '+' in hide:
				print('Protip: Use the plot_diff_var() function ya goober')
				self.range_index = (1, 2, 1)
			elif '-' in hide and '=' in hide:
				self.range_index = (2, 3, 1)
			elif '+' in hide and '=' in hide:
				self.range_index = (0, 1, 1)

		self.fig = plt.figure(figsize = (15,10))

		#self.fig.suptitle('Almeida 8 ODE Model' + model_version, fontsize = 20)
		self.ax1 = self.fig.add_subplot(1, 1, 1)
		#self.ax1.set_title('Effect of ' + param_name + ' perturbation on ' + var_name, fontsize = 19, pad = 10)

		for i in range(self.range_index[0], self.range_index[1], self.range_index[2]):

			self.var_props = Variable_Properties(self.solution_list[i], self.name_list, var_name, normalization, stability, peak_prominence)
			self.ax1.plot(self.solution_list[i].t, self.var_props.normed_solution[self.name_index_dict[var_name]], c = 'blue', linestyle = 'solid', linewidth = 1.5, label = self.lab[i])
			self.max_ex.append(self.var_props.max_val)
			self.max_ex.append(self.var_props.min_val)

		self.ax1.legend(loc = 'upper right',fontsize = 15)
		self.ax1.set_ylabel(self.y_tit + ' Expression [a.u.]', fontsize = 30)
		self.ax1.set_xlabel('Time [h]', fontsize = 30)
		self.ax1.tick_params('both', which = 'major', labelsize = 30)
		self.ax1.tick_params('both', which = 'minor', labelsize = 30)
		self.ax1.set_yticks(np.linspace(0, max(self.max_ex), max(self.max_ex)+1), minor = 'true')
		self.ax1.set_yticks(np.linspace(0, 50, 51), minor = 'true')
		self.ax1.set_xticks([i for i in range(time_range[0], time_range[1] + 1, 6)], minor = 'true')
		self.ax1.set_xticks([i for i in range(time_range[0], time_range[1], 24)])

		# self.test = []
		# self.minor_test = []
		self.label_test = []
		#for j in range(0, 1000):
		for i in range(0, 9877, 24):
			self.label_test.append(i)
		for i in range(0, 121, 24):
			self.label_test.append(i)

		# for j in range(0, 10000, 6):
		# 	self.test.append(j)
		
		# for k in range(0, 10000):
		# 	self.minor_test.append(k)

		# self.ax1.set_xticks(self.minor_test, minor = 'true')
		# self.ax1.set_xticks(self.test)
		self.ax1.set_xticklabels(self.label_test)
		#self.ax1.set_xticks([i for i in range(0, 100, 6)], minor = 'true')
		#self.ax1.set_xticks([i for i in range(time_range[0], time_range[1], 24)])
		
		self.lim_up = max(self.max_ex) + (max(self.max_ex)/10)
		self.lim_down = min(self.max_ex) - (min(self.max_ex)/10)
		if self.lim_down < 0:
			self.lim_down = 0

		self.ax1.set_ylim([self.lim_down, self.lim_up])
		self.ax1.set_xlim(time_range)
		#self.ax1.grid(which = 'major', color = 'grey', linestyle = '-', linewidth = 0.75)
		#self.ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')

		#plt.tight_layout()
		#plt.show()
		plt.savefig(self.directory + folder + param_name + '_effect_on_' + var_name + '.png')
		plt.close(self.fig)

	#Plot phase space of var_1 against var_2
	def plot_phase_space(self, var_1, var_2, folder, model_version = '', mod = None, 
						 normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		
		self.param_index = 0
		self.mod_index = 1
		self.title_add = ''
		self.file_suff = ''
		self.var_1_index = self.name_index_dict[var_1]
		self.var_2_index = self.name_index_dict[var_2]
		
		if mod != None:
			self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
			if mod[1] == '-':
				self.mod_index = 0
				self.title_add = '(' + mod[0] + ' - ' + str(self.modulation * 100) + '%' + ')'
				self.file_suff = mod[0] + '-' + str(self.modulation)
			elif mod[1] == '+':
				self.mod_index = 2
				self.title_add = '(' + mod[0] + ' + ' + str(self.modulation * 100) + '%' + ')'
				self.file_suff = mod[0] + '+' + str(self.modulation)

		self.solution = self.organized_solutions[self.mod_index][self.param_index]
		self.var_1_props = Variable_Properties(self.solution, self.name_list, var_1, normalization, stability, peak_prominence)
		self.var_2_props = Variable_Properties(self.solution, self.name_list, var_2, normalization, stability, peak_prominence)
		self.max_ex = []
		self.y_tit = ''
		if normalization == 'mean':
			self.y_tit = 'Mean-Normalized '
		elif normalization == 'max':
			self.y_tit = 'Max-Normalized '
		elif normalization == 'none':
			self.y_tit = ''

		self.fig = plt.figure(figsize = (15,10))

		self.fig.suptitle('Almeida 8 ODE Model' + model_version, fontsize = 20)
		self.ax1 = self.fig.add_subplot(1, 1, 1)
		self.ax1.set_title('Phase Space: ' + var_1 + ' - ' + var_2 + ' ' + self.title_add, fontsize = 19, pad = 20)

		self.ax1.plot(self.var_1_props.normed_solution[self.var_1_index], self.var_2_props.normed_solution[self.var_2_index], c = 'FireBrick', linestyle = 'solid', linewidth = 0.5, label = '')
		# self.max_ex.append(max(self.var_props.normed_solution[self.var_1_index]))
		# self.max_ex.append(max(self.var_props.normed_solution[self.var_2_index]))
		self.max_x = max(self.var_1_props.normed_solution[self.var_1_index])
		self.max_y = max(self.var_2_props.normed_solution[self.var_2_index])

		#self.ax1.legend(loc = 'upper right',fontsize = 15)
		self.ax1.set_ylabel(self.y_tit + var_2 + ' Expression [a.u.]', fontsize = 20)
		self.ax1.set_xlabel(self.y_tit + var_1 + ' Expression [a.u.]', fontsize = 20)
		self.ax1.tick_params('both', which = 'major', labelsize = 20)
		self.ax1.tick_params('both', which = 'minor', labelsize = 20)
		#self.ax1.set_yticks(np.linspace(0, max(self.max_ex), max(self.max_ex)+1), minor = 'true')
		#self.ax1.set_xticks(np.linspace(0, 20*24, (20*24/6)+1), minor = 'true')
		#self.ax1.set_xticks([i for i in range(time_range[0], time_range[1], 24)])
		self.ax1.set_ylim([0, self.max_y + self.max_y/10])
		self.ax1.set_xlim([0, self.max_x + self.max_x/10])
		#self.ax1.grid(which = 'major', color = 'grey', linestyle = '-', linewidth = 0.75)
		#self.ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')

		#plt.tight_layout()
		#plt.show()
		plt.savefig(self.directory + folder + var_1 + ' - ' + var_2 + '_phase-space' + self.file_suff + '.png')
		plt.close(self.fig)

	#Plot Factor Analysis Data
	def plot_fac(self, folder, file_pre, var_1, var_2, factor_range, model_property, greek = 'alpha', model_version = '', mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		
		self.min_max_list = []

		self.y_tit = ''
		if normalization == 'mean':
			self.y_tit = 'Mean-Normalized '
		elif normalization == 'max':
			self.y_tit = 'Max-Normalized '
		elif normalization == 'none':
			self.y_tit = ''

		self.greek = None
		if greek == 'alpha':
			self.greek = r'$\alpha$'
		elif greek == 'beta':
			self.greek = r'$\beta$'
		elif greek == 'gamma':
			self.greek = r'$\gamma$'

		# self.x_vals = [i for i in range(factor_range[0], factor_range[1], factor_range[2])]
		self.x_vals = factor_range

		self.fig = plt.figure(figsize = (15,10))

		self.fig.suptitle('Almeida 8 ODE Model' + model_version, fontsize = 20)
		self.ax1 = self.fig.add_subplot(1, 1, 1)

		if model_property == 'Period':

			self.ax1.set_title('Period Length against Inhibition Factor ' + self.greek, fontsize = 19, pad = 10)
			for i in range(0, len(self.name_list)):
				self.y_vals_period = Factor_Analysis(self.directory, folder, file_pre, factor_range).get_periods(self.name_list, self.name_list[i], mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
				self.ax1.scatter(self.x_vals, self.y_vals_period, c = self.sub_colors[i], s = 30, label = self.name_list[i] + ' Period')
				self.ax1.plot(self.x_vals, self.y_vals_period, c = self.sub_colors[i], linestyle = 'solid', linewidth = 1, alpha = 0.4, label = '')
				self.min_max_list.append(max(self.y_vals_period))
				self.min_max_list.append(min(self.y_vals_period))
			self.ax1.set_ylabel('Time [h]', fontsize = 20)
			self.ax1.legend(loc = 'upper left',fontsize = 14)
			self.file_suff = ''

			#Period Doubling 
			# self.ax1.set_title('Period Length against Inhibition Factor ' + self.greek, fontsize = 19, pad = 10)
			# for i in range(0, len(self.name_list)):
			# 	self.y_vals_period = Factor_Analysis(self.directory, folder, file_pre, factor_range).get_dub_periods(self.name_list, self.name_list[i], mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
			# 	self.ax1.scatter(self.x_vals, np.array(self.y_vals_period)/2, c = self.sub_colors[i], s = 30, label = self.name_list[i] + ' Period')
			# 	self.ax1.plot(self.x_vals, np.array(self.y_vals_period)/2, c = self.sub_colors[i], linestyle = 'solid', linewidth = 1, alpha = 0.4, label = '')
			# 	self.min_max_list.append(max(self.y_vals_period)/2)
			# 	self.min_max_list.append(min(self.y_vals_period)/2)
			# self.ax1.set_ylabel('Time [h]', fontsize = 20)
			# self.ax1.legend(loc = 'upper left',fontsize = 14)
			# self.file_suff = ''

		elif model_property == 'Delay':

			self.y_vals_delay = Factor_Analysis(self.directory, folder, file_pre, factor_range).get_delays(self.name_list, var_1, var_2, mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)

			self.ax1.set_title(var_1 + ' - ' + var_2 + ' Delay against Inhibition Factor ' + self.greek, fontsize = 19, pad = 10)
			self.ax1.scatter(self.x_vals, self.y_vals_delay, c = 'red', s = 30, label = 'Data Points')
			self.ax1.plot(self.x_vals, self.y_vals_delay, c = 'FireBrick', linestyle = 'solid', linewidth = 1, alpha = 0.5, label = '')
			self.min_max_list.append(max(self.y_vals_delay))
			self.min_max_list.append(min(self.y_vals_delay))
			self.ax1.set_ylabel(self.y_tit + var_1 + ' - ' + var_2 + ' Delay [h]', fontsize = 20)
			self.ax1.legend(loc = 'upper left',fontsize = 14)
			self.file_suff = ''

		elif model_property == 'Magnitude':
			
			self.y_vals_amp = Factor_Analysis(self.directory, folder, file_pre, factor_range).get_all_peak_mags(self.name_list, var_1, mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
			self.ax1.set_title(var_1 + ' Peak Magnitudes against Inhibition Factor ' + self.greek, fontsize = 19, pad = 10)
			for i in range(0, len(self.x_vals)):
				if i == 0:
					self.ax1.scatter([self.x_vals[i] for j in range(0, len(self.y_vals_amp[i]))], self.y_vals_amp[i], c = self.tri_colore[0], s = 30, alpha = 0.3, label = 'All Magnitudes')
					self.ax1.scatter([self.x_vals[i] for j in range(0, 2)], self.y_vals_amp[i][-2:], c = 'firebrick', s = 10, label = 'Last 2 Peaks')
				else:
					self.ax1.scatter([self.x_vals[i] for j in range(0, len(self.y_vals_amp[i]))], self.y_vals_amp[i], c = self.tri_colore[0], s = 30, alpha = 0.3)
					self.ax1.scatter([self.x_vals[i] for j in range(0, 2)], self.y_vals_amp[i][-2:], c = 'firebrick', s = 10)
				self.min_max_list.append(max(self.y_vals_amp[i]) + max(self.y_vals_amp[i])/50)
				self.min_max_list.append(min(self.y_vals_amp[i]) - min(self.y_vals_amp[i])/50)
			self.ax1.set_ylabel(self.y_tit + var_1 + ' Peak Magnitudes [a.u.]', fontsize = 20)
			self.ax1.legend(loc = 'upper left',fontsize = 14)
			self.file_suff = '-' + var_1 

		elif model_property == 'Amplitude':
			
			self.y_vals_amp = Factor_Analysis(self.directory, folder, file_pre, factor_range).get_all_peak_mags(self.name_list, var_1, amp = True, mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
			self.ax1.set_title(var_1 + ' Peak Amplitudes against Inhibition Factor ' + self.greek, fontsize = 19, pad = 10)
			for i in range(0, len(self.x_vals)):
				if i == 0:
					self.ax1.scatter([self.x_vals[i] for j in range(0, len(self.y_vals_amp[i]))], self.y_vals_amp[i], c = self.tri_colore[0], s = 30, alpha = 0.3, label = 'All Magnitudes')
					self.ax1.scatter([self.x_vals[i] for j in range(0, 2)], self.y_vals_amp[i][-2:], c = 'firebrick', s = 10, label = 'Last 2 Peaks')
				else:
					self.ax1.scatter([self.x_vals[i] for j in range(0, len(self.y_vals_amp[i]))], self.y_vals_amp[i], c = self.tri_colore[0], s = 30, alpha = 0.3)
					self.ax1.scatter([self.x_vals[i] for j in range(0, 2)], self.y_vals_amp[i][-2:], c = 'firebrick', s = 10)
				self.min_max_list.append(max(self.y_vals_amp[i]) + max(self.y_vals_amp[i])/50)
				self.min_max_list.append(min(self.y_vals_amp[i]) - min(self.y_vals_amp[i])/50)
			self.ax1.set_ylabel(self.y_tit + var_1 + ' Peak Amplitudes [a.u.]', fontsize = 20)
			self.ax1.legend(loc = 'upper left',fontsize = 14)
			self.file_suff = '-' + var_1 + '_up'

		
		self.ax1.set_xlabel('Inhibition Factor '+ self.greek, fontsize = 20)
		self.ax1.tick_params('both', which = 'major', labelsize = 20)
		self.ax1.tick_params('both', which = 'minor', labelsize = 20)
		# self.ax1.set_yticks([i for i in range(0, 19)], minor = 'true')
		#self.ax1.set_xticks(np.linspace(0, 20*24, (20*24/6)+1), minor = 'true')
		self.ax1.set_xticks([i/1000 for i in range(0, 101, 10)])
		self.ax1.set_ylim([min(self.min_max_list)-(min(self.min_max_list)/50), max(self.min_max_list)+(max(self.min_max_list)/50)])
		self.ax1.set_xlim([-0.005, 0.105])
		#self.ax1.grid(which = 'major', color = 'grey', linestyle = '-', linewidth = 0.75)
		#self.ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')

		#plt.tight_layout()
		#plt.show()
		plt.savefig(self.directory + 'fac_plot_' + model_property + self.file_suff + '.png')
		plt.close(self.fig)

	#Plot Fourier Analysis
	def plot_fourier(self, var_1, folder = '', title = '', model_version = '', mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):

		self.param_index = 0
		self.mod_index = 1
		self.title_add = ''
		self.file_suff = ''
		self.var_1_index = self.name_index_dict[var_1]
		
		if mod != None:
			self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
			if mod[1] == '-':
				self.mod_index = 0
				self.title_add = '(' + mod[0] + ' - ' + str(self.modulation * 100) + '%' + ')'
				self.file_suff = mod[0] + '-' + str(self.modulation)
			elif mod[1] == '+':
				self.mod_index = 2
				self.title_add = '(' + mod[0] + ' + ' + str(self.modulation * 100) + '%' + ')'
				self.file_suff = mod[0] + '+' + str(self.modulation)

		self.solution = self.organized_solutions[self.mod_index][self.param_index]
		self.var_props = Variable_Properties(self.solution, self.name_list, var_1, normalization, stability, peak_prominence)
		self.y_tit = ''
		if normalization == 'mean':
			self.y_tit = 'Mean-Normalized '
		elif normalization == 'max':
			self.y_tit = 'Max-Normalized '
		elif normalization == 'none':
			self.y_tit = ''

		self.normed_sol = self.var_props.normed_solution[self.var_1_index]
		self.simtime = self.modulate_parameters_object.time_span[1] - self.modulate_parameters_object.time_span[0]
		self.N = len(self.normed_sol)
		self.T = self.simtime/self.N
		#self.T = 1/self.simtime
		self.xf = fft.fftfreq(self.N, self.T)
		self.yf = fft.fft(self.normed_sol)
		#self.yf = self.yfr/max(self.yfr)
		# self.nf = 1/(2 * self.T)
		self.nf = self.T/2

		self.fig = plt.figure(figsize = (15,10))

		self.fig.suptitle('Almeida 8 ODE Model' + model_version, fontsize = 20)
		self.ax1 = self.fig.add_subplot(1, 1, 1)
		self.ax1.set_title('Power Spectrum: ' + title , fontsize = 19, pad = 10)

		self.ax1.plot(self.xf, self.yf, c = 'Blue', linestyle = 'solid', linewidth = 1, label = var_1 + ' ' + self.title_add)

		self.ax1.legend(loc = 'upper right',fontsize = 15)
		self.ax1.set_ylabel('Amplitude [a.u.]', fontsize = 20)
		self.ax1.set_xlabel('Frequency [Hz]', fontsize = 20)
		self.ax1.tick_params('both', which = 'major', labelsize = 20)
		self.ax1.tick_params('both', which = 'minor', labelsize = 20)
		#self.ax1.set_yticks(np.linspace(0, max(self.max_ex), max(self.max_ex)+1), minor = 'true')
		#self.ax1.set_xticks(np.linspace(0, 20*24, (20*24/6)+1), minor = 'true')
		#self.ax1.set_xticks([i for i in range(time_range[0], time_range[1], 24)])
		#self.ax1.set_ylim([0, max(self.yf)+(max(self.yf)/10)])
		self.ax1.set_yscale('log')
		self.ax1.set_xlim([0, self.nf])
		#self.ax1.grid(which = 'major', color = 'grey', linestyle = '-', linewidth = 0.75)
		#self.ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')

		#plt.tight_layout()
		#plt.show()
		plt.savefig(self.directory + folder + var_1 + '_power-spectrum_' + self.file_suff + '.png')
		plt.close(self.fig)

	#Plot all Peak Magnitudes and Corresponding Timestamps
	def plot_peak_data(self, var_1, folder, peak_range = None, model_version = '', mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):

		self.param_index = 0
		self.mod_index = 1
		self.title_add = ''
		self.file_suff = ''
		self.var_1_index = self.name_index_dict[var_1]
		
		# self.suptitle_add = ''
		# if model_version == 'standard':
		# 	self.suptitle_add = ' - Standard Configuration'
		# elif model_version == 'competetive':
		# 	self.suptitle_add = ' - Competetive Inhibition'
		# elif model_version == 'independent':
		# 	self.suptitle_add = ' - Independent Inhibition'

		if mod != None:
			self.param_index = self.modulate_parameters_object.param_index_dict[mod[0]]
			if mod[1] == '-':
				self.mod_index = 0
				self.title_add = '(' + mod[0] + ' - ' + str(self.modulation * 100) + '%' + ')'
				self.file_suff = mod[0] + '-' + str(self.modulation)
			elif mod[1] == '+':
				self.mod_index = 2
				self.title_add = '(' + mod[0] + ' + ' + str(self.modulation * 100) + '%' + ')'
				self.file_suff = mod[0] + '+' + str(self.modulation)

		self.y_tit = ''
		if normalization == 'mean':
			self.y_tit = 'Mean-Normalized '
		elif normalization == 'max':
			self.y_tit = 'Max-Normalized '
		elif normalization == 'none':
			self.y_tit = ''

		self.solution = self.organized_solutions[self.mod_index][self.param_index]
		self.var_props = Variable_Properties(self.solution, self.name_list, var_1, normalization, stability, peak_prominence)
		self.peak_data = self.var_props.get_all_peak_data()
		self.peak_count = [i for i in range(1, len(self.peak_data[0])+1)]
		self.peak_range = [0-5, len(self.peak_count)+5]

		if peak_range != None and type(peak_range) == list or type(peak_range) == tuple:
			self.peak_range = peak_range 

		#Plot Variable Amplitude/Period Control Analysis
		self.fig = plt.figure(figsize = (15,10))
		self.fig.suptitle('Almeida 8 ODE Model' + model_version, fontsize = 20)

		self.ax1 = self.fig.add_subplot(2, 1, 1)
		self.ax1.set_title('Peak Magnitudes' + ' [' + var_1 + ']', fontsize = 19, pad = 10)

		self.ax1.scatter(self.peak_count, self.peak_data[0], s = 5, c = self.strong_tri_colore[0], marker = '^', label = 'Peaks')
		self.ax1.plot(self.peak_count, self.peak_data[0], c = self.tri_colore[0], linestyle = 'solid', alpha = 0.5, linewidth = 1)

		self.ax1.legend(loc = 'upper right',fontsize = 15)
		self.ax1.set_ylabel(self.y_tit + 'Magnitude [a.u.]', fontsize = 18)
		self.ax1.tick_params('both', which = 'major', labelsize = 13)
		#self.ax1.tick_params('both', which = 'minor', labelsize = 10)
		#self.ax1.set_yticks(np.linspace(0, 5, 11), minor = 'true')
		self.ax1.set_ylim([min(self.peak_data[0]) - min(self.peak_data[0])/100, max(self.peak_data[0]) + max(self.peak_data[0])/100])
		self.ax1.set_xlim(self.peak_range)
		plt.setp(self.ax1.get_xticklabels(), visible=False)
		self.ax1.grid(which = 'major', axis = 'x', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')

		self.ax2 = self.fig.add_subplot(2, 1, 2)
		self.ax2.set_title('Period Lengths' + ' [' + var_1 + ']', fontsize = 19, pad = 10)

		self.ax2.scatter(self.peak_count[1:], self.peak_data[1], s = 5, c = self.strong_tri_colore[2], marker = 'o', label = 'Periods')
		self.ax2.plot(self.peak_count[1:], self.peak_data[1], c = self.tri_colore[2], linestyle = 'solid', alpha = 0.5, linewidth = 1)

		self.ax2.legend(loc = 'upper right',fontsize = 15)
		self.ax2.set_ylabel('Period Length [h]', fontsize = 18)
		self.ax2.set_xlabel(r'n$_{peaks}$', fontsize = 18, labelpad = 15)
		self.ax2.tick_params('both', which = 'major', labelsize = 13)
		#self.ax2.tick_params('both', which = 'minor', labelsize = 10)
		#self.ax2.set_yticks(np.linspace(0, 5, 11), minor = 'true')
		self.ax2.set_ylim([min(self.peak_data[1]) - min(self.peak_data[1])/100, max(self.peak_data[1]) + max(self.peak_data[1])/100])
		self.ax2.set_xlim(self.peak_range)
		self.ax2.grid(which = 'major', axis = 'x', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax2.set_facecolor('whitesmoke')

		plt.savefig(self.directory + folder + 'all_peaks_' + var_1 + self.file_suff + '.png')
		plt.close(self.fig)

	#Plot Mod Analysis Data for single parameter graduated modulation
	def plot_single_param_data(self, file_pre, param_name, var_1, var_2, mod_range, model_property, export_dir = None, greek = 'alpha', model_version = '', mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		
		self.min_max_list = []

		self.y_tit = ''
		if normalization == 'mean':
			self.y_tit = 'Mean-Normalized '
		elif normalization == 'max':
			self.y_tit = 'Max-Normalized '
		elif normalization == 'none':
			self.y_tit = ''

		self.greek = None
		if greek == 'alpha':
			self.greek = r'$\alpha$'
		elif greek == 'beta':
			self.greek = r'$\beta$'
		elif greek == 'gamma':
			self.greek = r'$\gamma$'

		# self.x_vals = [i for i in range(factor_range[0], factor_range[1], factor_range[2])]
		self.x_vals = mod_range

		self.fig = plt.figure(figsize = (15,10))

		self.fig.suptitle('Almeida 8 ODE Model' + model_version, fontsize = 20)
		self.ax1 = self.fig.add_subplot(1, 1, 1)

		if model_property == 'Period':

			self.ax1.set_title('Period Length against ' + param_name + ' Perturbation', fontsize = 19, pad = 10)
			for i in range(0, len(self.name_list)):
				self.y_vals_period = Mod_Analysis(self.directory, file_pre, mod_range).get_periods(self.name_list, self.name_list[i], mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
				self.ax1.scatter(self.x_vals, self.y_vals_period, c = self.sub_colors[i], s = 30, label = self.name_list[i] + ' Period')
				self.ax1.plot(self.x_vals, self.y_vals_period, c = self.sub_colors[i], linestyle = 'solid', linewidth = 1, alpha = 0.4, label = '')
				self.min_max_list.append(max(self.y_vals_period))
				self.min_max_list.append(min(self.y_vals_period))
			self.ax1.set_ylabel('Peak-Peak Difference [h]', fontsize = 20)
			self.ax1.legend(loc = 'upper left',fontsize = 14)
			self.file_suff = ''

			#Period Doubling 
			# self.ax1.set_title('Period Length against Inhibition Factor ' + self.greek, fontsize = 19, pad = 10)
			# for i in range(0, len(self.name_list)):
			# 	self.y_vals_period = Mod_Analysis(self.directory, file_pre, mod_range).get_dub_periods(self.name_list, self.name_list[i], mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
			# 	self.ax1.scatter(self.x_vals, np.array(self.y_vals_period)/2, c = self.sub_colors[i], s = 30, label = self.name_list[i] + ' Period')
			# 	self.ax1.plot(self.x_vals, np.array(self.y_vals_period)/2, c = self.sub_colors[i], linestyle = 'solid', linewidth = 1, alpha = 0.4, label = '')
			# 	self.min_max_list.append(max(self.y_vals_period)/2)
			# 	self.min_max_list.append(min(self.y_vals_period)/2)
			# self.ax1.set_ylabel('Time [h]', fontsize = 20)
			# self.ax1.legend(loc = 'upper left',fontsize = 14)
			# self.file_suff = ''

		elif model_property == 'Delay':

			self.y_vals_delay = Mod_Analysis(self.directory, file_pre, mod_range).get_delays(self.name_list, var_1, var_2, mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)

			self.ax1.set_title(var_1 + ' - ' + var_2 + ' Delay against ' + param_name + ' Perturbation', fontsize = 19, pad = 10)
			self.ax1.scatter(self.x_vals, self.y_vals_delay, c = 'red', s = 30, label = 'Data Points')
			self.ax1.plot(self.x_vals, self.y_vals_delay, c = 'FireBrick', linestyle = 'solid', linewidth = 1, alpha = 0.5, label = '')
			self.min_max_list.append(max(self.y_vals_delay))
			self.min_max_list.append(min(self.y_vals_delay))
			self.ax1.set_ylabel(self.y_tit + var_1 + ' - ' + var_2 + ' Delay [h]', fontsize = 20)
			self.ax1.legend(loc = 'upper left',fontsize = 14)
			self.file_suff = ''

		elif model_property == 'Magnitude':
			
			self.y_vals_amp = Mod_Analysis(self.directory, file_pre, mod_range).get_all_peak_mags(self.name_list, var_1, mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
			self.ax1.set_title(var_1 + ' Peak Magnitudes against ' + param_name + ' Perturbation', fontsize = 19, pad = 10)
			for i in range(0, len(self.x_vals)):
				if i == 0:
					self.ax1.scatter([self.x_vals[i] for j in range(0, len(self.y_vals_amp[i]))], self.y_vals_amp[i], c = self.tri_colore[0], s = 30, alpha = 0.3, label = 'All Magnitudes')
					self.ax1.scatter([self.x_vals[i] for j in range(0, 2)], self.y_vals_amp[i][-2:], c = 'firebrick', s = 10, label = 'Last 2 Peaks')
				else:
					self.ax1.scatter([self.x_vals[i] for j in range(0, len(self.y_vals_amp[i]))], self.y_vals_amp[i], c = self.tri_colore[0], s = 30, alpha = 0.3)
					self.ax1.scatter([self.x_vals[i] for j in range(0, 2)], self.y_vals_amp[i][-2:], c = 'firebrick', s = 10)
				self.min_max_list.append(max(self.y_vals_amp[i]) + max(self.y_vals_amp[i])/50)
				self.min_max_list.append(min(self.y_vals_amp[i]) - min(self.y_vals_amp[i])/50)
			self.ax1.set_ylabel(self.y_tit + var_1 + ' Peak Magnitudes [a.u.]', fontsize = 20)
			self.ax1.legend(loc = 'upper left',fontsize = 14)
			self.file_suff = '-' + var_1 

		elif model_property == 'Amplitude':
			
			# self.y_vals_amp = Mod_Analysis(self.directory, file_pre, mod_range).get_all_peak_mags(self.name_list, var_1, amp = True, mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
			# self.ax1.set_title(var_1 + ' Peak Amplitudes against Inhibition Factor ' + self.greek, fontsize = 19, pad = 10)
			# for i in range(0, len(self.x_vals)):
			# 	if i == 0:
			# 		self.ax1.scatter([self.x_vals[i] for j in range(0, len(self.y_vals_amp[i]))], self.y_vals_amp[i], c = self.tri_colore[0], s = 30, alpha = 0.3, label = 'All Magnitudes')
			# 		self.ax1.scatter([self.x_vals[i] for j in range(0, 2)], self.y_vals_amp[i][-2:], c = 'firebrick', s = 10, label = 'Last 2 Peaks')
			# 	else:
			# 		self.ax1.scatter([self.x_vals[i] for j in range(0, len(self.y_vals_amp[i]))], self.y_vals_amp[i], c = self.tri_colore[0], s = 30, alpha = 0.3)
			# 		self.ax1.scatter([self.x_vals[i] for j in range(0, 2)], self.y_vals_amp[i][-2:], c = 'firebrick', s = 10)
			# 	self.min_max_list.append(max(self.y_vals_amp[i]) + max(self.y_vals_amp[i])/50)
			# 	self.min_max_list.append(min(self.y_vals_amp[i]) - min(self.y_vals_amp[i])/50)
			# self.ax1.set_ylabel(self.y_tit + var_1 + ' Peak Amplitudes [a.u.]', fontsize = 20)
			# self.ax1.legend(loc = 'upper left',fontsize = 14)
			# self.file_suff = '-' + var_1 + '_up'

			self.ax1.set_title('Amplitude against ' + param_name + ' Perturbation', fontsize = 19, pad = 10)
			for i in range(0, len(self.name_list)):
				self.y_vals_amplitude = Mod_Analysis(self.directory, file_pre, mod_range).get_amplitudes(self.name_list, self.name_list[i], mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
				self.ax1.scatter(self.x_vals, self.y_vals_amplitude, c = self.sub_colors[i], s = 30, label = self.name_list[i] + ' Amplitude')
				self.ax1.plot(self.x_vals, self.y_vals_amplitude, c = self.sub_colors[i], linestyle = 'solid', linewidth = 1, alpha = 0.4, label = '')
				self.min_max_list.append(max(self.y_vals_amplitude))
				self.min_max_list.append(min(self.y_vals_amplitude))
			self.ax1.set_ylabel('Peak Amplitudes [a.u.]', fontsize = 20)
			self.ax1.legend(loc = 'upper left',fontsize = 14)
			self.file_suff = ''
		
		self.ax1.set_xlabel('Parameter Perturbation Factor', fontsize = 20)
		self.ax1.tick_params('both', which = 'major', labelsize = 20)
		self.ax1.tick_params('both', which = 'minor', labelsize = 20)
		# self.ax1.set_yticks([i for i in range(0, 19)], minor = 'true')
		self.ax1.set_xticks([i for i in mod_range], minor = 'true')
		#self.ax1.set_xticks([i/1000 for i in range(0, 101, 10)])
		self.ax1.set_ylim([min(self.min_max_list)-(min(self.min_max_list)/50), max(self.min_max_list)+(max(self.min_max_list)/50)])
		self.ax1.set_xlim([mod_range[0] - 0.02, mod_range[-1] + 0.02])
		#self.ax1.grid(which = 'major', color = 'grey', linestyle = '-', linewidth = 0.75)
		#self.ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')

		#plt.tight_layout()
		#plt.show()
		if export_dir != None:
			plt.savefig(export_dir + 'mod_plot_' + param_name + '_' + model_property + self.file_suff + '.png')
		else:
			plt.savefig(self.directory + 'mod-plot_' + param_name + '_' + model_property + self.file_suff + '.png')

		plt.close(self.fig)

	#Plot Mod Analysis Data for multiple paramter graduated modulations
	def plot_param_delays(self, file_pre, param_name, var_1, var_2, mod_range, export_dir = None, period = False, period_norm = False, greek = 'alpha', model_version = '', mod = None, normalization = 'mean', stability = (0.99, 1.01), peak_prominence = 0.5):
		
		self.min_max_list = []

		self.y_tit = ''
		if normalization == 'mean':
			self.y_tit = 'Mean-Normalized '
		elif normalization == 'max':
			self.y_tit = 'Max-Normalized '
		elif normalization == 'none':
			self.y_tit = ''

		self.greek = None
		if greek == 'alpha':
			self.greek = r'$\alpha$'
		elif greek == 'beta':
			self.greek = r'$\beta$'
		elif greek == 'gamma':
			self.greek = r'$\gamma$'

		# self.x_vals = [i for i in range(factor_range[0], factor_range[1], factor_range[2])]
		self.x_vals = mod_range

		self.fig = plt.figure(figsize = (15,10))

		self.fig.suptitle('Almeida 8 ODE Model' + model_version, fontsize = 20)
		self.ax1 = self.fig.add_subplot(1, 1, 1)

		self.ax1.set_title(var_1 + ' - ' + var_2 + ' Delay against Degradation Rate Perturbation', fontsize = 19, pad = 10)
		
		if period == False:
			for i in range(0, len(param_name)):

				self.y_vals_delay = Mod_Analysis(self.directory + param_name[i] + '/', file_pre + param_name[i] + '_mod-', mod_range).get_delays(self.name_list, var_1, var_2, mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)

				if period_norm == True:
					self.ax1.set_title(var_1 + ' - ' + var_2 + ' Delay against Degradation Rate Perturbation (Normed to Period Length)', fontsize = 19, pad = 10)
					self.delay_array = np.array(self.y_vals_delay)
					self.period_array = np.array(Mod_Analysis(self.directory + param_name[i] + '/', file_pre + param_name[i] + '_mod-', mod_range).get_periods(self.name_list, 'BMAL1', mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence))
					self.ax1.scatter(self.x_vals, self.delay_array/self.period_array * 24.8, c = self.sub_colors[i], s = 30, label = param_name[i])
					self.ax1.plot(self.x_vals, self.delay_array/self.period_array * 24.8, c = self.sub_colors[i], linestyle = 'solid', linewidth = 1, alpha = 0.5, label = '')
					self.file_suff = 'period_renormed'
					self.min_max_list.append(max(self.delay_array/self.period_array * 24.8))
					self.min_max_list.append(min(self.delay_array/self.period_array * 24.8))
					self.ax1.set_ylabel(var_1 + ' - ' + var_2 + ' Delay Normed to Period Length [h]', fontsize = 20)

				else:
					self.ax1.scatter(self.x_vals, self.y_vals_delay, c = self.sub_colors[i], s = 30, label = param_name[i])
					self.ax1.plot(self.x_vals, self.y_vals_delay, c = self.sub_colors[i], linestyle = 'solid', linewidth = 1, alpha = 0.5, label = '')
					self.file_suff = ''
					self.min_max_list.append(max(self.y_vals_delay))
					self.min_max_list.append(min(self.y_vals_delay))
					self.ax1.set_ylabel(self.y_tit + var_1 + ' - ' + var_2 + ' Delay [h]', fontsize = 20)

				self.ax1.legend(loc = 'upper center',fontsize = 14)

		if period == True:
			self.ax1.set_title('Period Length against Degradation Rate Perturbation', fontsize = 19, pad = 10)
			for j in range(0, len(param_name)):
				for i in range(0, len(self.name_list)):
					self.y_vals_period = Mod_Analysis(self.directory + param_name[j] + '/', file_pre + param_name[j] + '_mod-', mod_range).get_periods(self.name_list, self.name_list[i], mod = mod, normalization = normalization, stability = stability, peak_prominence = peak_prominence)
					if i == 7:
						self.ax1.scatter(self.x_vals, self.y_vals_period, c = self.sub_colors[j], s = 30, label = param_name[j])
						self.ax1.plot(self.x_vals, self.y_vals_period, c = self.sub_colors[j], linestyle = 'solid', linewidth = 1, alpha = 0.4, label = '')
						print('i = 7: ', self.sub_colors[j])
					else:
						# self.ax1.scatter(self.x_vals, self.y_vals_period, c = self.sub_colors[i], s = 30, label = '')
						# self.ax1.plot(self.x_vals, self.y_vals_period, c = self.sub_colors[i], linestyle = 'solid', linewidth = 1, alpha = 0.4, label = '')
						print('i = ' + str(i) + ': ', self.sub_colors[i])
					self.min_max_list.append(max(self.y_vals_period))
					self.min_max_list.append(min(self.y_vals_period))
			self.ax1.set_ylabel('Peak-Peak Difference [h]', fontsize = 20)
			self.ax1.legend(loc = 'upper center',fontsize = 14)
			self.file_suff = 'periods'
		
		self.ax1.set_xlabel('Parameter Perturbation Factor', fontsize = 20)
		self.ax1.tick_params('both', which = 'major', labelsize = 20)
		self.ax1.tick_params('both', which = 'minor', labelsize = 20)
		# self.ax1.set_yticks([i for i in range(0, 19)], minor = 'true')
		self.ax1.set_xticks([i for i in mod_range], minor = 'true')
		#self.ax1.set_xticks([i/1000 for i in range(0, 101, 10)])
		self.ax1.set_ylim([min(self.min_max_list)-(min(self.min_max_list)/50), max(self.min_max_list)+(max(self.min_max_list)/50)])
		self.ax1.set_xlim([mod_range[0] - 0.02, mod_range[-1] + 0.02])
		#self.ax1.grid(which = 'major', color = 'grey', linestyle = '-', linewidth = 0.75)
		#self.ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
		self.ax1.set_facecolor('whitesmoke')

		#plt.tight_layout()
		#plt.show()
		if export_dir != None:
			plt.savefig(export_dir + 'mod_plot_all-deg_' + self.file_suff + '.png')
		else:
			plt.savefig(self.directory + 'mod-plot_all-deg_' + self.file_suff + '.png')

		plt.close(self.fig)


#----------------------------------------------------------------------------------------------------------------------------------------------------

#Check Korencic Paper!!! 

#trig Fit
#get stable times
#choose later one
#if statement (if one returns -1, choose other)
#get slice of both solution.y
#apply fit

#sin fit test from stackoverflow --> not very good :'-(

# def fit_sin(tt, yy):
#     '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
#     tt = np.array(tt)
#     yy = np.array(yy)
#     ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
#     Fyy = abs(np.fft.fft(yy))
#     guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
#     guess_amp = np.std(yy) * 2.**0.5
#     guess_offset = np.mean(yy)
#     guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

#     def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
#     popt, pcov = op.curve_fit(sinfunc, tt, yy, p0=guess)
#     A, w, p, c = popt
#     f = w/(2.*np.pi)
#     fitfunc = lambda t: A * np.sin(w*t + p) + c
#     return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


# test = fit_sin(solution.t, solution.y[6])

# plt.plot(solution.t, test['fitfunc'](solution.t))
# plt.plot(solution.t, solution.y[6])
# plt.show()

# class test:
# 	def __init__(self, a):
# 		self.a = a

# 	def testerino(self, b):
# 		self.say = b
# 		print(self.say)
# 		print(self.say, self.a)


# for j in range(0, len(names)):
# 	amp_test = MAM.Control_Analysis(modulated_solutions, 'Amplitude', names, names[j]).property_data
# 	per_test = MAM.Control_Analysis(modulated_solutions, 'Period', names, names[j]).property_data
# 	amp_base = amp_test[1][0]
# 	per_base = per_test[1][0]
# 	amp_lims = new_lims(amp_test)
# 	per_lims = new_lims(per_test)


	
# 	#Plot Variable Amplitude/Period Control Analysis
# 	fig = plt.figure(figsize = (15,10))
# 	fig.suptitle('Almeida 8 ODE model - Control Analysis [' + names[j] + ']', fontsize = 20)

# 	ax1 = fig.add_subplot(2, 1, 1)
# 	ax1.set_title('Amplitude', fontsize = 19)

# 	for i in range(2, -1, -2):
# 		ax1.scatter(am.param_names, amp_test[i], s = 30, c = strong_tri_colore[i], marker = markerinos[i], label = labelinos[i] + str(modulation * 100) + '%')
# 		ax1.plot(am.param_names, amp_test[i], c = tri_colore[i], linestyle = 'solid', alpha = 0.5, linewidth = 1)
# 	ax1.plot(am.param_names, amp_test[1], c = tri_colore[1], linestyle = 'solid', alpha = 1, linewidth = 1, label = labelinos[1])

# 	ax1.legend(loc = 'upper right',fontsize = 15)
# 	ax1.set_ylabel('Norm. Amplitude [a.u.]', fontsize = 18)
# 	#ax1.set_xlabel('Time [h]', fontsize = 25)
# 	ax1.tick_params('both', which = 'major', labelsize = 13)
# 	#ax1.tick_params('both', which = 'minor', labelsize = 10)
# 	#ax1.set_yticks(np.linspace(0, 5, 11), minor = 'true')
# 	#ax1.set_xticks(np.linspace(0, 168, 29), minor = 'true')
# 	#ax1.set_xticks([i for i in range(0, 200, 24)])
# 	ax1.set_ylim(amp_lims)
# 	#ax1.set_xlim(time_span)
# 	plt.setp(ax1.get_xticklabels(), visible=False)
# 	ax1.grid(which = 'major', axis = 'x', color = 'grey', linestyle = '-', linewidth = 0.5)
# 	#ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
# 	ax1.set_facecolor('whitesmoke')
# 	ax1.add_patch(pat.Rectangle((0, amp_base*0.99), len(am.param_names)-1, (amp_base * 1.01 - amp_base * 0.99), alpha = 0.2, color = 'green'))
# 	ax1.add_patch(pat.Rectangle((0, amp_base*0.97), len(am.param_names)-1, (amp_base * 1.03 - amp_base * 0.97), alpha = 0.1, color = 'green'))
# 	ax1.add_patch(pat.Rectangle((0, amp_base*0.95), len(am.param_names)-1, (amp_base * 1.05 - amp_base * 0.95), alpha = 0.05, color = 'green'))
# 	ax1.add_patch(pat.Rectangle((0, amp_base*0.90), len(am.param_names)-1, (amp_base * 1.10 - amp_base * 0.90), alpha = 0.035, color = 'green'))

# 	ax2 = fig.add_subplot(2, 1, 2)
# 	ax2.set_title('Period', fontsize = 19)

# 	for i in range(2, -1, -2):
# 		ax2.scatter(am.param_names, per_test[i], s = 30, c = strong_tri_colore[i], marker = markerinos[i], label = labelinos[i] + str(modulation * 100) + '%')
# 		ax2.plot(am.param_names, per_test[i], c = tri_colore[i], linestyle = 'solid', alpha = 0.5, linewidth = 1)
# 	ax2.plot(am.param_names, per_test[1], c = tri_colore[1], linestyle = 'solid', alpha = 1, linewidth = 1, label = labelinos[1])

# 	ax2.set_ylabel('Period Length [h]', fontsize = 18)
# 	ax2.set_xlabel('Model Parameters', fontsize = 18, labelpad = 15)
# 	ax2.tick_params('both', which = 'major', labelsize = 13)
# 	#ax2.tick_params('both', which = 'minor', labelsize = 20)
# 	#ax2.set_yticks(np.linspace(0, 12, 13), minor = 'true')
# 	#ax2.set_xticks(np.linspace(0, 168, 29), minor = 'true')
# 	#ax2.set_xticks([i for i in range(0, 200, 24)])
# 	ax2.set_ylim(per_lims)
# 	#ax2.set_xlim(time_span)
# 	ax2.grid(which = 'major', axis = 'x', color = 'grey', linestyle = '-', linewidth = 0.5)
# 	#ax2.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
# 	ax2.set_facecolor('whitesmoke')
# 	ax2.add_patch(pat.Rectangle((0, per_base*0.99), len(am.param_names)-1, (per_base * 1.01 - per_base * 0.99), alpha = 0.2, color = 'green'))
# 	ax2.add_patch(pat.Rectangle((0, per_base*0.97), len(am.param_names)-1, (per_base * 1.03 - per_base * 0.97), alpha = 0.1, color = 'green'))
# 	ax2.add_patch(pat.Rectangle((0, per_base*0.95), len(am.param_names)-1, (per_base * 1.05 - per_base * 0.95), alpha = 0.05, color = 'green'))
# 	ax2.add_patch(pat.Rectangle((0, per_base*0.90), len(am.param_names)-1, (per_base * 1.10 - per_base * 0.90), alpha = 0.035, color = 'green'))

# 	#plt.tight_layout()
# 	plt.savefig(direct + '/test/control_' + names[j] + '.png')


# #Plot PER-Cry Delay
# fig = plt.figure(figsize = (15,10))
# fig.suptitle('Almeida 8 ODE model - Control Analysis [PER - CRY]', fontsize = 20)

# ax1 = fig.add_subplot(1, 1, 1)
# ax1.set_title('Delay', fontsize = 19, pad = 10)

# for i in range(2, -1, -2):
# 	ax1.scatter(am.param_names, per_cry_delay[i], s = 30, c = strong_tri_colore[i], marker = markerinos[i], label = labelinos[i] + str(modulation * 100) + '%')
# 	ax1.plot(am.param_names, per_cry_delay[i], c = tri_colore[i], linestyle = 'solid', alpha = 0.5, linewidth = 1)
# ax1.plot(am.param_names, per_cry_delay[1], c = tri_colore[1], linestyle = 'solid', alpha = 1, linewidth = 1, label = labelinos[1])

# ax1.legend(loc = 'upper right',fontsize = 15)
# ax1.set_ylabel('Peak Delay [h]', fontsize = 18)
# ax1.set_xlabel('Model Parameters', fontsize = 18, labelpad = 15)
# ax1.tick_params('both', which = 'major', labelsize = 13)
# #ax1.tick_params('both', which = 'minor', labelsize = 10)
# #ax1.set_yticks(np.linspace(0, 5, 11), minor = 'true')
# #ax1.set_xticks(np.linspace(0, 168, 29), minor = 'true')
# #ax1.set_xticks([i for i in range(0, 200, 24)])
# ax1.set_ylim(delay_lims)
# #ax1.set_xlim(time_span)
# #plt.setp(ax1.get_xticklabels(), visible=False)
# ax1.grid(which = 'major', axis = 'x', color = 'grey', linestyle = '-', linewidth = 0.5)
# #ax1.grid(which = 'minor', color = 'grey', linestyle = '-', linewidth = 0.5)
# ax1.set_facecolor('whitesmoke')
# ax1.add_patch(pat.Rectangle((0, baser*0.99), len(am.param_names)-1, (baser * 1.01 - baser * 0.99), alpha = 0.2, color = 'green'))
# ax1.add_patch(pat.Rectangle((0, baser*0.97), len(am.param_names)-1, (baser * 1.03 - baser * 0.97), alpha = 0.1, color = 'green'))
# ax1.add_patch(pat.Rectangle((0, baser*0.95), len(am.param_names)-1, (baser * 1.05 - baser * 0.95), alpha = 0.05, color = 'green'))
# ax1.add_patch(pat.Rectangle((0, baser*0.90), len(am.param_names)-1, (baser * 1.10 - baser * 0.90), alpha = 0.035, color = 'green'))
# plt.savefig(direct + '/test/control_' + 'per-cry_delay' + '.png')

