
#Almeida 8 ODE Circadian Clock Model
#Implemented by Camillo Mizaikoff @ 2020

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Import Libraries

import numpy as np

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Static Values

#Parameters
V_R = 44.4
k_R = 3.54
k_Rr = 80.1
V_E = 30.3
k_E = 214
k_Er = 1.24
V_D = 202
k_D = 5.32
k_Dr = 94.7
G_ror = 2.55
G_rev = 0.4
G_p = 0.844
G_c = 2.34
G_db = 0.156
G_E4 = 0.295
G_pc = 0.19
G_cp = 0.141
G_bp = 2.58 

fac = 1

param_var_name_list = [V_R, k_R, k_Rr, V_E, k_E, k_Er, V_D, k_D, k_Dr, G_ror, G_rev, G_p, G_c, G_db, G_E4, G_pc, G_cp, G_bp]
param_names = ['V_R', 'k_R', 'k_Rr', 'V_E', 'k_E', 'k_Er', 'V_D', 'k_D', 'k_Dr', 'G_ror', 'G_rev', 'G_p', 'G_c', 'G_db', 'G_E4', 'G_pc', 'G_cp', 'G_bp']
param_val_list = [44.4, 3.54, 80.1, 30.3, 214, 1.24, 202, 5.32, 94.7, 2.55, 0.4, 0.844, 2.34, 0.156, 0.295, 0.19, 0.141, 2.58]
param_dict = {}
param_index_dict = {}
for i in range(0, len(param_names)):
	param_dict[param_names[i]] = param_val_list[i]
	param_index_dict[param_names[i]] = i

#Initial values
BMAL1_0 = 4
ROR_0 = 6
REV_0 = 2
DBP_0 = 1
E4BP4_0 = 4
CRY_0 = 1
PER_0 = 1
PER_CRY_0 = 1

init_vals = [BMAL1_0, ROR_0, REV_0, DBP_0, E4BP4_0, CRY_0, PER_0, PER_CRY_0]

var_names = ['BMAL1', 'ROR', 'REV', 'DBP', 'E4BP4', 'CRY', 'PER', 'PER:CRY']

#----------------------------------------------------------------------------------------------------------------------------------------------------
#Functions

#Almeida 8 ODE System
def almeida_circadian_clock(t, r):

	BMAL1 = r[0]
	ROR = r[1]
	REV = r[2]
	DBP = r[3]
	E4BP4 = r[4]
	CRY = r[5]
	PER = r[6]
	PER_CRY = r[7]

	V_R_m = param_val_list[param_index_dict['V_R']]
	k_R_m = param_val_list[param_index_dict['k_R']]
	k_Rr_m = param_val_list[param_index_dict['k_Rr']]
	V_E_m = param_val_list[param_index_dict['V_E']]
	k_E_m = param_val_list[param_index_dict['k_E']]
	k_Er_m = param_val_list[param_index_dict['k_Er']]
	V_D_m = param_val_list[param_index_dict['V_D']]
	k_D_m = param_val_list[param_index_dict['k_D']]
	k_Dr_m = param_val_list[param_index_dict['k_Dr']]
	G_ror_m = param_val_list[param_index_dict['G_ror']]
	G_rev_m = param_val_list[param_index_dict['G_rev']]
	G_p_m = param_val_list[param_index_dict['G_p']]
	G_c_m = param_val_list[param_index_dict['G_c']]
	G_db_m = param_val_list[param_index_dict['G_db']]
	G_E4_m = param_val_list[param_index_dict['G_E4']]
	G_pc_m = param_val_list[param_index_dict['G_pc']]
	G_cp_m = param_val_list[param_index_dict['G_cp']]
	G_bp_m = param_val_list[param_index_dict['G_bp']]

	# Old
	# E_box = ((V_E_m * BMAL1 * (1 - fac * BMAL1 * PER_CRY))/(BMAL1 + k_E_m + k_Er_m * BMAL1 * CRY))	
	# Without Control Term
	# E_box = ((V_E_m * BMAL1 * (1 - fac * PER_CRY))/(BMAL1 + k_E_m + k_Er_m * BMAL1 * CRY))
	# With Control Term
	E_box = ((V_E_m * BMAL1 * (1 - fac * PER_CRY))/(BMAL1 * (1 - fac * PER_CRY) + k_E_m + k_Er_m * BMAL1 * CRY))
	
	RRE = (V_R_m * ROR * k_Rr_m**2)/((ROR + k_R_m) * (k_Rr_m**2 + REV**2))
	D_box = (V_D_m * DBP * k_Dr_m)/((DBP + k_D_m) * (k_Dr_m + E4BP4))

	dBMAL1 = RRE - G_bp_m * BMAL1 * PER_CRY
	dROR = E_box + RRE - G_ror_m * ROR
	#dROR = 0
	dREV = 2 * E_box + D_box - G_rev_m * REV
	dDBP = E_box - G_db_m * DBP
	dE4BP4 = 2 * RRE - G_E4_m * E4BP4
	#dE4BP4 = 0
	dCRY = E_box + 2 * RRE - G_pc_m * PER * CRY + G_cp_m * PER_CRY - G_c_m * CRY
	#dCRY = 0
	dPER = E_box + D_box - G_pc_m * PER * CRY + G_cp_m * PER_CRY - G_p_m * PER
	#dPER = 0
	dPER_CRY = G_pc_m * PER * CRY - G_cp_m * PER_CRY - G_bp_m * BMAL1 * PER_CRY

	return [dBMAL1, dROR, dREV, dDBP, dE4BP4, dCRY, dPER, dPER_CRY]


#----------------------------------------------------------------------------------------------------------------------------------------------------
