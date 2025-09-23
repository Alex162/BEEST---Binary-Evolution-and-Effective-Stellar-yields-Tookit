import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob as glob
import copy
import numba as nb
import warnings
import time as time
from scipy.optimize import curve_fit
import re
import os
import radioactivedecay as rd
from mendeleev import isotope
from decimal import Decimal
import importlib as il

import important_functions as im
il.reload(im)
warnings.filterwarnings("ignore")


def make_table(savetag='',savepath='',h=0.5,beta_winds=0.1,beta_rlof=1,beta_cc=0.00,accurate_radio=True,
	list_radio=['Al-26','Fe-60','Mn-53','Cl-36','Ca-41','Pd-205']):
	tinit=time.time()
	savename_abs=f"{savepath}h={h}_betawinds={beta_winds}_betarlof={beta_rlof}_betacc={beta_cc}{savetag}_abs.txt"
	savename_net=f"{savepath}h={h}_betawinds={beta_winds}_betarlof={beta_rlof}_betacc={beta_cc}{savetag}_net.txt"

	isolist=im.make_iso_list_rob(im.gridpath)
	binary_df,single_df=im.read_iso_rob('h1')
	binary_df_tab34,single_df_tab34=im.read_tab34_rob()

	init_abu_dict,init_abu_arr=im.calc_init_abu_dict()
	out_abu_dict,out_abu_arr=im.calc_out_abu_dict()
	# net_abu_dict,net_abu_arr=im.calc_net_abu_dict()

	init_fractions=init_abu_arr/np.sum(out_abu_arr,axis=0)
	init_mass=init_fractions * np.array(binary_df['m_init'])
	init_mass_dict={}
	for i, keyname in enumerate(isolist):
		init_mass_dict[keyname]=init_mass[i,:]


	print('calculating effective yields')


	dM,_=im.calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc)
	dumping_arr,_=im.calc_dumping_arr(beta_winds=beta_winds,beta_rlof=beta_rlof,
                                         beta_cc=beta_cc)

	if accurate_radio:
	    converted_elements = [im.convert_format(el) for el in isolist]
	    inv_dict1 = im.create_inventory(dumping_arr, converted_elements, list_radio=list_radio)
	    inv_dict_new1 = im.create_new_inventory(dumping_arr, converted_elements, list_radio)
	    mean_timing_radio_list = list(map(lambda Mpriminit: im.calc_avg_timing(Mpriminit)[0], binary_df['m_init']))

	    dumping_arr,_ = im.tej_func(mean_timing_radio_list, dumping_arr, inv_dict1,inv_dict_new1,converted_elements=converted_elements)


	table_list_abs=[]
	table_list_net=[]
	for i,iso in enumerate(isolist):
		y_abs=im.calc_Yeff_abs(iso,h=h,dM=dM,dumping_arr_iso=dumping_arr[np.where(isolist==iso)].flatten(),
                    beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc)
		y_net=y_abs-init_mass_dict[iso]
		table_list_abs.append(y_abs)
		table_list_net.append(y_net)


	print(time.time()-tinit)


	print("creating ZAtuplist and yeff_dict")
	yeff_dict_abs={}
	yeff_dict_net={}
	ZAtuplist=[]
	for i,_ in enumerate(table_list_abs):
		keyname=isolist[i]
		yeff_dict_abs[keyname]=table_list_abs[i]
		yeff_dict_net[keyname]=table_list_net[i]

		Z,A=im.getZ_A(keyname)
		ZAtup=(keyname,Z,A)
		ZAtuplist.append(ZAtup)

	ZAtuplist_sorted=sorted(ZAtuplist, key = lambda sub: (sub[1], sub[2]))
	
	pretty_iso_sorted=[]
	for i,tup in enumerate(ZAtuplist_sorted):
		pretty_iso_sorted.append(im.pretty_iso(tup[0]))


	print(f'writing abs table to: {savename_abs}')
	print(f'writing net table to: {savename_net}')
	print(time.time()-tinit)
	fabs=open(savename_abs,'w')
	im.write_preamble(fabs,h,beta_winds,beta_rlof,beta_cc,pretty_iso_list=pretty_iso_sorted)
	fabs.close()

	fnet=open(savename_net,'w')
	im.write_preamble(fnet,h,beta_winds,beta_rlof,beta_cc,pretty_iso_list=pretty_iso_sorted)
	fnet.close()
	

	for massnum, Mprim in enumerate(binary_df['m_init']):
		Mrem=Mprim-np.sum(np.array(table_list_abs)[:,massnum])
		fabs=open(savename_abs,'a')
		im.write_star_header(fabs,Minit=Mprim,Mfinal=Mrem,lifetime=np.array(binary_df_tab34['Age'])[massnum])
		fabs.close()#if you don't open/close it will not write part of the Hydrogen string for some reason.

		fnet=open(savename_net,'a')
		im.write_star_header(fnet,Minit=Mprim,Mfinal=Mrem,lifetime=np.array(binary_df_tab34['Age'])[massnum])
		fnet.close()#if you don't open/close it will not write part of the Hydrogen string for some reason.

		fabs=open(savename_abs,'a')
		fnet=open(savename_net,'a')
		for i,tup in enumerate(ZAtuplist_sorted):
			if tup[0]=='neut':
				continue

			init_abu_abs=init_mass_dict[tup[0]][massnum]
			yieldnum_abs=yeff_dict_abs[tup[0]][massnum]

			init_abu_net=0
			yieldnum_net=yeff_dict_net[tup[0]][massnum]

			isostring='&'+im.pretty_iso(tup[0]).ljust(9, ' ')
			Zstr='&'+str(tup[1]).ljust(3, ' ')
			Astr='&'+str(tup[2]).ljust(3, ' ')

			im.write_star_data(fabs,tup[0],yieldnum_abs,init_abu_abs,tup[1],tup[2])
			im.write_star_data(fnet,tup[0],yieldnum_net,init_abu_net,tup[1],tup[2])
		fabs.close()
		fnet.close()
	print(f'done writing abs table to: {savename_abs}')
	print(f'done writing net table to: {savename_net}')
	print(time.time()-tinit)

