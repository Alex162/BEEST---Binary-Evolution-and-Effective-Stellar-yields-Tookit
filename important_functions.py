#important functions
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

###REPLACE THIS WITH YOUR RELEVANT PATH TO CODE
codepath='/Users/alexkemp/Desktop/postdoc_stuff/binary_yields_stuff/github/BEEST---Binary-Evolution-and-Effective-Stellar-yields-Tookit/'
###

gridpath=codepath+'tables/grid/'
binary_c_singles_path=codepath
table34_path=codepath+'tables/'


gridpath='/Users/alexkemp/Desktop/postdoc_stuff/binary_yields_stuff/tables/grid/'
binary_c_singles_path='/Users/alexkemp/Desktop/postdoc_stuff/binary_yields_stuff/'
table34_path='/Users/alexkemp/Desktop/postdoc_stuff/binary_yields_stuff/tables/'


class EquationCalculator:
    def __init__(self, Ysingle, Yprim, Ysec, h, q):
        self.Ysingle = Ysingle
        self.Yprim = Yprim
        self.Ysec = Ysec
        self.h = h
        self.q = q

    def calculate(self):
        numerator = (1 - self.h) * self.Ysingle + self.h * (self.Yprim + self.Ysec)
        denominator = 1 + self.h * self.q
        result = numerator / denominator
        return result
    

def tej_func(decay_times, dumping_arr, inv_dict1,inv_zero_slr,converted_elements):
    #todo: better name for invdict1
    final_yields_dict = {}
    

    # decayed inventory for each star and isotope
    for star_key, star_data in inv_zero_slr.items():
        if star_key not in final_yields_dict:
            final_yields_dict[star_key] = {}
        
        # Add the original yields to final_yields_dict
        for isotope, yield_value in star_data.items():
            final_yields_dict[star_key][isotope] = yield_value

    for i, (star_key, star_data) in enumerate(inv_dict1.items()):
        inv1_update = rd.Inventory(star_data, 'g')
        decay_time = decay_times[i]
        inv1_update_decayed = inv1_update.decay(decay_time, 'My')

        inv_t1_masses = inv1_update_decayed.masses('g')

        if star_key not in final_yields_dict:
            final_yields_dict[star_key] = {}

        for isotope, decayed_mass in inv_t1_masses.items():
            if isotope in final_yields_dict[star_key]:
                final_yields_dict[star_key][isotope] += decayed_mass

    # 2D array of yields
    final_yields_2D_array = np.array([
        [final_yields.get(isotope, 0) for isotope in converted_elements]
        for final_yields in final_yields_dict.values()
    ]).T  # Transpose for correct shape
    return final_yields_2D_array, final_yields_dict



def convert_format(element):
    match = re.match(r"([a-zA-Z]+)(\d+)", element)
    if match:
        prefix = match.group(1)
        number = match.group(2)
        return f"{prefix.capitalize()}-{number}"
    return element


# list_radio=['Al-26','Fe-60','Mn-53','Cl-36','Ca-41','Pd-205']

def create_inventory(dumping_arr, isolist, list_radio=['Al-26','Fe-60','Mn-53','Cl-36','Ca-41','Pd-205']):
    inventory_dict = {}
    
    # Iterating over each star for all isotopes and yield data
    for star_index in range(dumping_arr.shape[1]):
        star_inventory = {}
        
        # Iterate over each isotope
        for isotope_index, isotope in enumerate(isolist):
            # Only consider isotopes that are in list_radio
            if isotope in list_radio:
                # Extract the mass value for the current isotope and star
                mass_value = dumping_arr[isotope_index, star_index]
                
                # If mass_value is negative, set it to zero
                if mass_value < 0:
                    print(isotope)
                    mass_value = 0
                    print('negative mass')
                
            
                if mass_value > 0:
                    star_inventory[isotope] = mass_value
        
        
        inventory_dict[f'star_{star_index}'] = star_inventory
    
    return inventory_dict


# data = pd.read_csv('/home/tejpreet/radioactivedecay/radioactivedecay/icrp107_ame2020_nubase2020/icrp.csv')#, allow_pickle=True)
 


def create_new_inventory(dumping_arr, isolist, list_radio=['Al-26','Fe-60','Mn-53','Cl-36','Ca-41','Pd-205']):
    inventory_dict_new = {}
    
  
    for star_index in range(dumping_arr.shape[1]):
        star_inventory_new = {}
        
        # Iterate over each isotope
        for isotope_index, isotope in enumerate(isolist):
            # Extract the mass value for the current isotope and star
            mass_value = dumping_arr[isotope_index, star_index]
            
            # If isotope is in list_radio, set its mass value to zero
            if isotope in list_radio:
                mass_value = 0
            
        
            # Add to star inventory
            star_inventory_new[isotope] = mass_value
        
        
        inventory_dict_new[f'star_{star_index}'] = star_inventory_new
    
    return inventory_dict_new



    
    
def calculate_Ysec(single_yields, q, mass_list):#assumption: mass_list is primary masses
    Ysec_values = []
    for mass in mass_list:
        Msec = mass * q# I think this is an issue. Want: average secondary yield. This is average secondary mass...
        if Msec <= 10:
            Ysec_values.append(0)
        else:
            sum_of_yields = sum(single_yields[i] for i in range(len(single_yields)) if mass_list[i] <= Msec)#... which is here used as an upper bound for the secondary yields considered.
            valid_mass_count = sum(1 for m in mass_list if m <= Msec)
            
            if valid_mass_count > 0:
                Ysec_values.append(sum_of_yields / valid_mass_count)
            else:
                Ysec_values.append(0)
    return Ysec_values


def find_between( s, first, last ):
    #https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]
    
def make_iso_list_rob(gridpath=gridpath):
    globarr=np.array(list(glob.glob(gridpath+'*_binary.dat')))
    globarr=np.sort(globarr)
    for i,string in enumerate(globarr):
        globarr[i]=find_between(string,gridpath,'_binary.dat')
        
        
    return globarr

def pretty_iso(iso):
    try:
        split=re.split(r'(\d+)',iso)
        pretty_iso=split[0].title()+'-'+split[1]
        return pretty_iso
    except:
        print("couldn't make '"+iso+"' pretty")
        return iso
    
def getZ_A(iso,robiso=True):
    if robiso:
        if iso=='neut':
            #neutrinos.
            return 0,0
        #will work for rob formatted iso names
        split=re.split(r'(\d+)',iso)
        element_str=split[0].title()
        A=int(split[1])
        
        Z=isotope(element_str,mass_number=A).atomic_number
        return Z,A
    
    else:
        #it had better look like 'Al-26'
        split=re.split('-',iso)
        element_str=split[0]
        A=int(split[1])
        
        Z=isotope(element_str,mass_number=A).atomic_number
        return Z,A
    

def read_binaryc_single_summary(path=binary_c_singles_path):
    binaryc_df=pd.read_csv(path+'binary_c_singles_summary.dat',sep=' ',index_col=False)
    #columns: 'M_init', 'age', 'core_mass', 'st'
    #core mass seems wrong for large initial masses, beware! seems ok for low mass stars though.
    binaryc_df=binaryc_df.sort_values('M_init')
    
    return binaryc_df

def read_tab34_rob(table34_path=table34_path,
                 nan_standin_binary=-1,nan_standin_single=-1):

    binary_filename=table34_path+'rob_table_4.dat'
    single_filename=table34_path+'rob_table_3.dat'
    
    

    binary_df_tab34=pd.read_csv(binary_filename,sep=' ',index_col=False)
    single_df_tab34=pd.read_csv(single_filename,sep=' ',index_col=False)
                          
    delete_me_binary= binary_df_tab34['Age']==nan_standin_binary
    binary_df_tab34=binary_df_tab34[~delete_me_binary]
    
    delete_me_single= single_df_tab34['Age']==nan_standin_single
    single_df_tab34=single_df_tab34[~delete_me_single]
   
    
    return binary_df_tab34,single_df_tab34
    
    
def read_iso_rob(iso,gridpath=gridpath,
                 nan_standin_binary=-1,nan_standin_single=-1,return_delete_me=False):
    
    #todo: supernova fudge/correction for Rob's (wrong) remnant masses
    binary_filename=gridpath+iso+'_binary.dat'
    single_filename=gridpath+iso+'_single.dat'
    
    binary_df=pd.read_csv(binary_filename,sep=' ',skiprows=1,
                          names=['m_init','winds','winds_init','rlof','rlof_init','cc','cc_init'])
    single_df=pd.read_csv(single_filename,sep=' ',skiprows=1,
                          names=['m_init','winds','winds_init','cc','cc_init'])
    
    delete_me_binary= binary_df['winds_init']==nan_standin_binary
    binary_df=binary_df[~delete_me_binary]
    
    delete_me_single= single_df['winds_init']==nan_standin_single
    single_df=single_df[~delete_me_single]
    if return_delete_me:
        return binary_df,single_df,delete_me_binary,delete_me_single
    
    return binary_df,single_df

def calc_Yprim_yields_abs(iso,beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath):
    binary_df,_=read_iso_rob(iso,gridpath)
#     print(binary_df)
    #absolute yield calculation
    Yprim=((1-beta_winds)*np.array(binary_df['winds']) + (1-beta_rlof)* np.array(binary_df['rlof']) + (1-beta_cc)*np.array(binary_df['cc']))
    
#     print('Yprim='+str(Yprim)+'\n')
    return Yprim

def calc_Ysingle_abs(iso,gridpath=gridpath,binary_size=False):
    if binary_size:
        _,single_df,delete_me_binary,_=read_iso_rob(iso,gridpath,return_delete_me=True)
    else:
        _,single_df=read_iso_rob(iso,gridpath)
    single_df=single_df[~delete_me_binary]
    
    Ysing=np.array(single_df['winds']) +np.array(single_df['cc'])
    
    return Ysing

def calc_Ysingle_net(iso,gridpath=gridpath,binary_size=False):
    if binary_size:
        _,single_df,delete_me_binary,_=read_iso_rob(iso,gridpath,return_delete_me=True)
    else:
        _,single_df=read_iso_rob(iso,gridpath)
    single_df=single_df[~delete_me_binary]
    
    Ysing=np.array(single_df['winds']) - np.array(single_df['winds_init']) +np.array(single_df['cc'])-np.array(single_df['cc_init'])
    
    return Ysing


def calc_Yprim_yields_net(iso,beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath):
    binary_df,_=read_iso_rob(iso,gridpath)
    # note: the init terms are the amount of mass of the iso initially present
    # in the amount of mass lost for a given mass loss regime. Thus it scales with the mass lost,
    # and so needs to be included in the beta product when considering the net yield!)
    Yprim=((1-beta_winds)*(binary_df['winds'] - binary_df['winds_init'])
                 +(1-beta_rlof)*(binary_df['rlof'] - binary_df['rlof_init'])
                 +(1-beta_cc)*(binary_df['cc'] - binary_df['cc_init']))
    return Yprim



def calc_dM(beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath):
    #returns:
#     dM under the given beta_winds and beta_rlof
#     An array summarising the breakdown for each star
    
    isolist=make_iso_list_rob(gridpath)
    
    #initiallise:
    binary_df,single_df=read_iso_rob(isolist[0],gridpath)
    len_binary_df=len(binary_df)
    dM=np.zeros(len_binary_df)
    dMwinds=np.zeros(len_binary_df)
    dMrlof=np.zeros(len_binary_df)
    dMcc=np.zeros(len_binary_df)

    for i,iso in enumerate(isolist):
        binary_df,single_df=read_iso_rob(iso,gridpath)

        #check length:
        if len(binary_df)==len_binary_df:
            pass
        else:
            print(iso)
            print('df size missmatch! I assume same size binary_df for each isotope!')
            raise a

        dM+=((beta_winds * binary_df['winds']) + (beta_rlof * binary_df['rlof']) + (beta_cc * binary_df['cc']))
            
        dMwinds+=binary_df['winds']
        dMrlof+=binary_df['rlof']
        dMcc+=binary_df['cc']

    #output of breakdown arr should be comparable with Table 4 Farmer et al 2023 and table 6 (Mfinal-Mrem)
    breakdownarr=np.array([np.array(dMwinds),np.array(dMrlof),np.array(dMcc)])
    
    return np.array(dM),breakdownarr
    

def calc_yields_avg_sec_abs(iso,dM='default',beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath):
    
    #returns: <Y_sec>, Y_prim (and other useful things) for a given value of beta(s)
    
    if type(dM)==str:
        dM,_=calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
    
    binary_df,single_df=read_iso_rob(iso,gridpath)
    
    primary_yield_abs=calc_Yprim_yields_abs(iso,beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
    
    #Singles (including a prepend for low mass extrapolation where necessary)
    
    abs_yields_single=list(single_df['winds'] + single_df['cc'])
    abs_yields_single_copy=copy.copy(abs_yields_single)
    # net_yields_single=list(single_df['winds']-single_df['winds_init'])
    lower_extrap_value=abs_yields_single[0]/single_df['m_init'][0]
#     print(single_df['m_init'][0])
#     print(abs_yields_single[0])
    abs_yields_single.insert(0, lower_extrap_value)#prepend the lower extrap value
#     abs_yields_single.insert(0, 0)#prepend the lower extrap value

    Minit_single=list(single_df['m_init'])
    Minit_single_copy=copy.copy(Minit_single)
    Minit_single.insert(0,1)#prepend the mass corresponding to the lower extrap value
    
    #Secondaries:
    Msec_list=[]
    interp_yields_sec_abs=[]
    avg_yield_sec_abs=[]
    for i,_ in enumerate(dM):
#         print(list(binary_df['m_ini'])[i])
        Msec_birth=np.arange(list(binary_df['m_init'])[i])#linearly spaced masses (dM=1 spacing) up to primary mass
        #TODO: make a more generalised q-depentent function so we can vary q a bit
        
        Msec_list.append(list(Msec_birth + dM[i]))#calculate all possible masses for secondaries after mass transfer
        
        if np.all(np.diff(Minit_single) > 0):
            #interpolate the yield for the masses in Msec_list ({A uniform distribution of secondary masses < Mprim}+dM(Mprim))
            interp_yields_sec_abs.append(np.interp(Msec_list[i],Minit_single,abs_yields_single))
            
        else:
            'Minit_single MUST be strictly increasing and have no nan'
    
        #calculate average secondary yield from the interpolated secondary yields:
        avg_yield_sec_abs.append(np.sum(interp_yields_sec_abs[i])/len(interp_yields_sec_abs[i]))

    return avg_yield_sec_abs,primary_yield_abs,dM,Msec_list,interp_yields_sec_abs,Minit_single_copy,abs_yields_single_copy,single_df,binary_df



    
def calc_yields_avg_sec_net(iso,dM='default',beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath):
    
    #returns: <Y_sec>, Y_prim (and other useful things) for a given value of beta(s)
    
    if type(dM)==str:
        dM,_=calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
    
    binary_df,single_df=read_iso_rob(iso,gridpath)
    
    primary_yield_net=calc_Yprim_yields_net(iso,beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
    
    #Singles (including a prepend for low mass extrapolation where necessary)
    lower_extrap_value=0
    net_yields_single=list(single_df['winds']-single_df['winds_init'] + single_df['cc']-single_df['cc_init'])
    net_yields_single_copy=copy.copy(net_yields_single)
    # net_yields_single=list(single_df['winds']-single_df['winds_init'])
    net_yields_single.insert(0, lower_extrap_value)#prepend the lower extrap value
    
    Minit_single=list(single_df['m_init'])
    Minit_single_copy=copy.copy(Minit_single)
    Minit_single.insert(0,0)#prepend the mass corresponding to the lower extrap value
    
    #secondaries:
    Msec_list=[]
    interp_yields_sec_net=[]
    avg_yield_sec_net=[]
    for i,_ in enumerate(dM):
#         print(list(binary_df['m_init'])[i])
        Msec_birth=np.arange(list(binary_df['m_init'])[i])#linearly spaced masses (dM=1 spacing) up to primary mass
        #TODO: make a more generalised q-depentent function so we can vary q a bit
        
        Msec_list.append(list(Msec_birth + dM[i]))#calculate all possible masses for secondaries after mass transfer
        
        if np.all(np.diff(Minit_single) > 0):
            #interpolate the yield for the masses in Msec_list ({A uniform distribution of secondary masses < Mprim}+dM(Mprim))
            interp_yields_sec_net.append(np.interp(Msec_list[i],Minit_single,net_yields_single))
            
        else:
            'Minit_single MUST be strictly increasing and have no nan'
    
        #calculate average secondary yield from the interpolated secondary yields:
        avg_yield_sec_net.append(np.sum(interp_yields_sec_net[i])/len(interp_yields_sec_net[i]))

    return avg_yield_sec_net,primary_yield_net,dM,Msec_list,interp_yields_sec_net,Minit_single_copy,net_yields_single_copy,single_df,binary_df


def calc_individual_binary_abs(iso,Mpriminit,Msecinit,beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath):
    
    dM,breakdownarr=calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)

    binary_df,single_df=read_iso_rob(iso,gridpath)
    
    #TODO: extrap treatment?
    dM_interp=np.interp(Mpriminit,binary_df['m_init'],dM)
    
    Minit_single=list(single_df['m_init'])
    Minit_single.insert(0,0)#prepend the mass corresponding to the lower extrap value
    
    abs_yields_single=list(single_df['winds'] + single_df['cc'])
    lower_extrap_value=0
    abs_yields_single.insert(0, lower_extrap_value)#prepend the lower extrap value
    
    Msec=Msecinit+dM_interp
    
    if np.all(np.diff(Minit_single) > 0):
        #interpolate the yield for the masses in Msec_list ({A uniform distribution of secondary masses < Mprim}+dM(Mprim))
        interp_yields_sec_abs = np.interp(Msec,Minit_single,abs_yields_single)
    else:
        'Minit_single MUST be strictly increasing and have no nan'
    
    
    primary_yields_abs=calc_Yprim_yields_abs(iso,beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
    primary_yield_abs = np.interp(Mpriminit,binary_df['m_init'],primary_yields_abs)
    
    
    return interp_yields_sec_abs,Msec,primary_yield_abs,Mpriminit
    
    

def calc_individual_binary_net(iso,Mpriminit,Msecinit,beta_winds=0,beta_rlof=1,beta_cc=0,
                               gridpath=gridpath):
    
    dM,breakdownarr=calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)

    binary_df,single_df=read_iso_rob(iso,gridpath)
    
    #TODO: extrap treatment?
    dM_interp=np.interp(Mpriminit,binary_df['m_init'],dM)
    
    Minit_single=list(single_df['m_init'])
    Minit_single.insert(0,0)#prepend the mass corresponding to the lower extrap value
    
    net_yields_single=list(single_df['winds']-single_df['winds_init'] + single_df['cc']-single_df['cc_init'])
    lower_extrap_value=0
    net_yields_single.insert(0, lower_extrap_value)#prepend the lower extrap value
    
    Msec=Msecinit+dM_interp
    
    if np.all(np.diff(Minit_single) > 0):
        #interpolate the yield for the masses in Msec_list ({A uniform distribution of secondary masses < Mprim}+dM(Mprim))
        interp_yields_sec_net = np.interp(Msec,Minit_single,net_yields_single)
    else:
        'Minit_single MUST be strictly increasing and have no nan'
    
    
    primary_yields_net=calc_Yprim_yields_net(iso,beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
    primary_yield_net = np.interp(Mpriminit,binary_df['m_init'],primary_yields_net)
    
    
    return interp_yields_sec_net,Msec,primary_yield_net,Mpriminit

def calc_Yeff_abs(iso,dumping_arr_iso=False,dM='default',h=0.5,q_avg=0.5,beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath,list_radio=['Al-26','Fe-60','Mn-53','Cl-36','Ca-41','Pd-205']):
    
    avg_yield_sec,primary_yield,dM,Msec_list,\
    interp_yields_sec,Minit_single,\
    yields_single,single_df,binary_df= calc_yields_avg_sec_abs(iso,dM=dM,
                                            beta_winds=beta_winds,beta_rlof=beta_rlof,
                                            beta_cc=beta_cc,gridpath=gridpath)
    
    
    if type(dumping_arr_iso)==bool:
        dumping_arr_iso=np.zeros(len(avg_yield_sec))
    
    avg_yield_sec=np.array(avg_yield_sec)+dumping_arr_iso
    
    
    yields_single_arr=np.array(yields_single)
    primary_yield_arr=np.array(primary_yield)
    Yeff_arr=np.zeros(primary_yield_arr.shape)
    
    for i,minit in enumerate(binary_df['m_init']):
#         print(minit)
#         print(np.array(single_df))
        s_index,_=find_nearest(np.array(single_df['m_init']),minit)
        
        calculator = EquationCalculator(yields_single_arr[s_index],primary_yield_arr[i],avg_yield_sec[i],h, q_avg)
        
        Yeff_arr[i] = calculator.calculate()
#         print('prim = ' + str(primary_yield_arr[i]))
#         print('sec = ' + str(avg_yield_sec[i]))
#         print('single = ' + str(net_yields_single_arr[s_index]))
#         print('Yeff = ' + str(Yeff_arr[i]))

#         print('\n')

    return Yeff_arr

def calc_Yeff_net(iso,dumping_arr_iso=False,dM='default',h=0.5,q_avg=0.5,beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath):
    
    avg_yield_sec,primary_yield,dM,Msec_list,\
    interp_yields_sec,Minit_single,\
    yields_single,single_df,binary_df= calc_yields_avg_sec_net(iso,dM=dM,
                                            beta_winds=beta_winds,beta_rlof=beta_rlof,
                                            beta_cc=beta_cc,gridpath=gridpath)
    
    
    if type(dumping_arr_iso)==bool:
        dumping_arr_iso=np.zeros(len(avg_yield_sec))
    
    avg_yield_sec=np.array(avg_yield_sec)+dumping_arr_iso
    
    
    yields_single_arr=np.array(yields_single)
    primary_yield_arr=np.array(primary_yield)
    Yeff_arr=np.zeros(primary_yield_arr.shape)
    
    for i,minit in enumerate(binary_df['m_init']):
#         print(minit)
#         print(np.array(single_df))
        s_index,_=find_nearest(np.array(single_df['m_init']),minit)
        
        calculator = EquationCalculator(yields_single_arr[s_index],primary_yield_arr[i],avg_yield_sec[i],h, q_avg)
        
        Yeff_arr[i] = calculator.calculate()
#         print('prim = ' + str(primary_yield_arr[i]))
#         print('sec = ' + str(avg_yield_sec[i]))
#         print('single = ' + str(net_yields_single_arr[s_index]))
#         print('Yeff = ' + str(Yeff_arr[i]))

#         print('\n')

    return Yeff_arr





def calc_dumping_arr(beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath,light_element_correction=True):
        #returns:
#     the net amount of each isotope dumped onto the secondary under given beta_winds, beta_rlof, and beta_cc.
# note that we ARE only net considering enrichment/depletion present in the accreted material here!
# Therefore you CANNOT add up the dumping array and expect to get the correct total mass transferred.
    isolist=make_iso_list_rob(gridpath)
    
    #initiallise:
    binary_df,single_df=read_iso_rob(isolist[0],gridpath)
    
    len_binary_df=len(binary_df)
    dumping_arr=[]

    for i,iso in enumerate(isolist):
#         print(iso)
        binary_df,single_df=read_iso_rob(iso,gridpath)

        #check length:
        if len(binary_df)==len_binary_df:
            pass
        else:
            print(iso)
            print('df size missmatch! I assume same size binary_df for each isotope!')
            raise a
    
    
        dumping_arr.append(np.array(((beta_winds * (binary_df['winds']-binary_df['winds_init'])) + 
             (beta_rlof * (binary_df['rlof']-binary_df['rlof_init'])) + 
             (beta_cc * ((binary_df['cc'])-binary_df['cc_init'])))))
        
    dumping_arr=np.array(dumping_arr)
        
    if light_element_correction:
        # Correction for the cases of fragile elements, assuming destruction/processing through H burning (as opposed to He burning):
        # 
        # Li-7 -> He4 (rapid decay of Be-8)
        # Be-7 (decays to Li-7 on 50d half life so treat as same) -> He4
        # Be-9 -> He3 and He-4, split by mass ratio of He-3 to 2 He-4 particles (really to Li-6, but Rob has no Li-6. And anyway, Li-6 also decays easily, but to He-3 and He-4 in equal measures.)
        # Be-10 -> He-4 (this thing is basically 0 anyway (i.e. no net yield to speak of), but do it just in case. Ultimately to He-4 through Li)
        # B-8 -> He-4 (unambiguous, its a rapid decay anyway meaning net yield should already be 0)
        # B-10 -> He-4 (two possible pathways, same result; they all pass through just other fragile isotopes (ultimately ending as He-4) or rapidly decaying ones into He-4)
        # B-11 -> He-4 (through rapid decay of Be-8)
        
        dumping_arr[np.where(isolist=='he4')]+=dumping_arr[np.where(isolist=='li7')]
        dumping_arr[np.where(isolist=='li7')]=0
        
        dumping_arr[np.where(isolist=='he4')]+=dumping_arr[np.where(isolist=='be7')]
        dumping_arr[np.where(isolist=='be7')]=0
        
        #be-9 into burns to 2 he4 + 1 he3, so 8/11 of the mass to he4 and 3/11 to he3.
        dumping_arr[np.where(isolist=='he4')]+= 0.72727272727*dumping_arr[np.where(isolist=='be9')]
        dumping_arr[np.where(isolist=='he3')]= 0.27272727273*dumping_arr[np.where(isolist=='be9')]
        dumping_arr[np.where(isolist=='be9')]=0
        
        dumping_arr[np.where(isolist=='he4')]+=dumping_arr[np.where(isolist=='be10')]
        dumping_arr[np.where(isolist=='be10')]=0
        
        dumping_arr[np.where(isolist=='he4')]+=dumping_arr[np.where(isolist=='b8')]
        dumping_arr[np.where(isolist=='b8')]=0
        
        dumping_arr[np.where(isolist=='he4')]+=dumping_arr[np.where(isolist=='b8')]
        dumping_arr[np.where(isolist=='b8')]=0
        
        dumping_arr[np.where(isolist=='he4')]+=dumping_arr[np.where(isolist=='b10')]
        dumping_arr[np.where(isolist=='b10')]=0
        
        dumping_arr[np.where(isolist=='he4')]+=dumping_arr[np.where(isolist=='b11')]
        dumping_arr[np.where(isolist=='b11')]=0
        
        
    
    return dumping_arr,isolist


def calc_dumping_arr_individual_binary(Mprim,beta_winds=0,beta_rlof=1,beta_cc=0,dumping_arr='default',gridpath=gridpath):
    #use this (does same thing as method 2, only faster)
#returns:
#     the net amount of each isotope dumped onto the secondary under given beta_winds, beta_rlof, and beta_cc.
# note that we ARE only net considering enrichment/depletion present in the accreted material here!
# Therefore you CANNOT add up the dumping array and expect to get the correct total mass transferred.
    
    if dumping_arr=='default':
        dumping_arr,isolist=calc_dumping_arr(beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
        
    
    
    #initiallise:
    binary_df,single_df=read_iso_rob(isolist[0],gridpath)
    
    dumping_arr_IB=[]
    
    for i,iso in enumerate(isolist):
         dumping_arr_IB.append(np.interp(Mprim,binary_df['m_init'],dumping_arr[i]))
    
    return np.array(dumping_arr_IB)

def He_core_fit_singles(M):
    #returns He core for singles according to a linear fit to Rob's data.
    #Note that this WILL NOT work at low M, as the y intercept is negative!
    # For low M, the binary_c singles should instead be used
    
    He_core_mass= -0.69540056 + 0.44313165*M
    
    return He_core_mass

def calc_individual_binary_timings_method2(Mpriminit,Msecinit,dM='default',binary_df_tab34='default',single_df_tab34='default',binaryc_singles='default',beta_winds=0,beta_rlof=1,beta_cc=0,gridpath=gridpath):
    #conserve He burnt (use this, not method 1)
    
    if (type(binary_df_tab34)!='pandas.core.frame.DataFrame'
         or type(single_df_tab34)!='pandas.core.frame.DataFrame'):
        binary_df_tab34,single_df_tab34=read_tab34_rob()
    
    if type(binaryc_singles)!='pandas.core.frame.DataFrame':
        binaryc_singles=read_binaryc_single_summary()
    
        
    ages_binary=np.array(binary_df_tab34['Age'])
    ages_single=np.array(single_df_tab34['Age'])
    
    if type(dM)==str:
        dM,breakdownarr=calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
    #TODO: extrap treatment?
    dM_interp=np.interp(Mpriminit,binary_df_tab34['Minit'],dM)
    
    Msec_after_dM=Msecinit+dM_interp
    
    Mcore_single_arr=np.array(single_df_tab34['He_core'])

    age_binary_prim=np.interp(Mpriminit,binary_df_tab34['Minit'],ages_binary)
    if Msecinit>=11:
        age_single_sec=np.interp(Msecinit,single_df_tab34['Minit'],ages_single)
    else:
        age_single_sec=np.interp(Msecinit,binaryc_singles['M_init'],binaryc_singles['age'])
 
    if Msecinit>=11:
        age_single_sec_after_dM=np.interp(Msec_after_dM,single_df_tab34['Minit'],ages_single)
    else:
        age_single_sec_after_dM=np.interp(Msec_after_dM,binaryc_singles['M_init'],binaryc_singles['age'])
    
    Mcore_single_sec=np.interp(Msecinit,single_df_tab34['Minit'],Mcore_single_arr)
    Mcore_single_sec_after_dM=np.interp(Msec_after_dM,single_df_tab34['Minit'],Mcore_single_arr)
    
    if Msecinit>=11:
        Mcore_single_sec=np.interp(Msecinit,single_df_tab34['Minit'],Mcore_single_arr)
#         Mcore_single_sec = He_core_fit_singles(Msecinit)
    else:
        Mcore_single_sec=np.interp(Msecinit,binaryc_singles['M_init'],binaryc_singles['core_mass'])
    
    if Msecinit>=11:
        Mcore_single_sec_after_dM=np.interp(Msec_after_dM,single_df_tab34['Minit'],Mcore_single_arr)
#         Mcore_single_sec = He_core_fit_singles(Msecinit)
    else:
        Mcore_single_sec_after_dM=np.interp(Msec_after_dM,binaryc_singles['M_init'],binaryc_singles['core_mass'])
    
    
    frac_single_life=age_binary_prim/age_single_sec
    He_proc_at_dM=frac_single_life*Mcore_single_sec # conserve this
    frac_sec_life_after_dM=He_proc_at_dM/Mcore_single_sec_after_dM
    
    decay_time=(1-frac_sec_life_after_dM)*age_single_sec_after_dM
    age_binary_sec=age_binary_prim+decay_time
    
    return age_binary_prim,age_single_sec,age_single_sec_after_dM,decay_time,age_binary_sec




def calc_avg_timing(Mpriminit,dM='default',
               binary_df_tab34='default',single_df_tab34='default',
                binaryc_singles='default',beta_winds=0,beta_rlof=1,beta_cc=0,
                gridpath=gridpath):
    
    
    if (type(binary_df_tab34)!='pandas.core.frame.DataFrame'
         or type(single_df_tab34)!='pandas.core.frame.DataFrame'):
        binary_df_tab34,single_df_tab34=read_tab34_rob()
        
    if type(dM)==str:
        dM,breakdownarr=calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath)
    #TODO: extrap treatment?
    dM_interp=np.interp(Mpriminit,binary_df_tab34['Minit'],dM)
    
    
    Msecinit=np.arange(Mpriminit)
    
    timing_arr=np.zeros(len(Msecinit))
    for i,_ in enumerate(Msecinit):
        temp_arr=calc_individual_binary_timings_method2(Mpriminit,Msecinit[i],
                        dM=dM,binary_df_tab34=binary_df_tab34,single_df_tab34=single_df_tab34,
                        binaryc_singles=binaryc_singles,beta_winds=beta_winds,beta_rlof=beta_rlof,
                        beta_cc=beta_cc,gridpath=gridpath)
        #age_binary_prim,age_single_sec,age_single_sec_after_dM,decay_time,age_binary_sec
        timing_arr[i]=temp_arr[3]
    
    mean_timing=np.mean(timing_arr)
    median_timing=np.median(timing_arr)
    
    return mean_timing,median_timing,timing_arr


def calc_Yeff_net_all(h=0.5,q_avg=0.5,beta_winds=0,beta_rlof=1,beta_cc=0,
              gridpath=gridpath,
                 accurate_radio=True,list_radio=['Al-26','Fe-60','Mn-53','Cl-36','Ca-41','Pd-205']):
    
    tinit=time.time()
    #calculate dumping arr and get isolist:
    dumping_arr,isolist=calc_dumping_arr(beta_winds=beta_winds,beta_rlof=beta_rlof,
                                         beta_cc=beta_cc,gridpath=gridpath)
    
    dM,_=calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,
                                         beta_cc=beta_cc,gridpath=gridpath)
    
    binary_df,single_df=read_iso_rob(isolist[0],gridpath)
    binary_df_tab34,single_df_tab34=read_tab34_rob()
    binaryc_singles=read_binaryc_single_summary()
    timing_arr=[]
    if accurate_radio:
        decay_times=np.zeros(len(np.array(binary_df['m_init'])))

        for i,Mpriminit in enumerate(binary_df['m_init']):
    #        temp_arr=[mean_timing,median_timing,timing_arr]
#             print(i)
#             temp_arr=calc_avg_timing(Mpriminit,dM=dM,
#                     binary_df_tab34=binary_df_tab34,single_df_tab34=single_df_tab34,
#                     binaryc_singles=binaryc_singles,beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,
#                     gridpath=gridpath)
#             #temp_arr=[mean_timing,median_timing,timingarr]
#             timing_arr.append(np.array(temp_arr[2]))
#             decay_times[i]=temp_arr[0] 
#         np.array(timing_arr)
#         print(decay_times)
#         print(timing_arr)
    
            converted_elements = [convert_format(el) for el in isolist]
            inv_dict1 = create_inventory(dumping_arr, converted_elements, list_radio=list_radio)
            inv_dict_new1 = create_new_inventory(dumping_arr, converted_elements, list_radio=list_radio)
            mean_timing_radio_list = list(map(lambda Mpriminit: calc_avg_timing(Mpriminit,dM=dM,
                            binary_df_tab34=binary_df_tab34,single_df_tab34=single_df_tab34,
                            binaryc_singles=binaryc_singles,beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,
                            gridpath=gridpath)[0], binary_df['m_init']))

            dumping_arr,_ = tej_func(mean_timing_radio_list, dumping_arr, inv_dict1,inv_dict_new1,converted_elements=converted_elements)

    
    Yeff_all=[]
    for i,iso in enumerate(isolist):
        dumping_arr_iso=dumping_arr[np.where(isolist==iso)].flatten()
        
        Yeff_all.append(calc_Yeff_net(iso,dM=dM,h=h,q_avg=q_avg,dumping_arr_iso=dumping_arr_iso,
                        beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath))
    print('elapsed = ' + str(time.time()-tinit))
    
    return Yeff_all



def calc_Yeff_abs_all(h=0.5,q_avg=0.5,beta_winds=0,beta_rlof=1,beta_cc=0,
            gridpath=gridpath,
            accurate_radio=True,list_radio=['Al-26','Fe-60','Mn-53','Cl-36','Ca-41','Pd-205']):
    
    tinit=time.time()
    #calculate dumping arr and get isolist:
    dumping_arr,isolist=calc_dumping_arr(beta_winds=beta_winds,beta_rlof=beta_rlof,
                                         beta_cc=beta_cc,gridpath=gridpath)
    
    dM,_=calc_dM(beta_winds=beta_winds,beta_rlof=beta_rlof,
                                         beta_cc=beta_cc,gridpath=gridpath)
    
    binary_df,single_df=read_iso_rob(isolist[0],gridpath)
    binary_df_tab34,single_df_tab34=read_tab34_rob()
    binaryc_singles=read_binaryc_single_summary()
    timing_arr=[]
    if accurate_radio:
        decay_times=np.zeros(len(np.array(binary_df['m_init'])))

        for i,Mpriminit in enumerate(binary_df['m_init']):
    #        temp_arr=[mean_timing,median_timing,timing_arr]
#             print(i)
#             temp_arr=calc_avg_timing(Mpriminit,dM=dM,
#                     binary_df_tab34=binary_df_tab34,single_df_tab34=single_df_tab34,
#                     binaryc_singles=binaryc_singles,beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,
#                     gridpath=gridpath)
#             #temp_arr=[mean_timing,median_timing,timingarr]
#             timing_arr.append(np.array(temp_arr[2]))
#             decay_times[i]=temp_arr[0] 
#         np.array(timing_arr)
#         print(decay_times)
#         print(timing_arr)
    #todo: check with tej that she no longer uses this; seems to work through the lambda thing now
    
            mean_timing_radio_list = list(map(lambda Mpriminit: calc_avg_timing(Mpriminit,dM=dM,
                            binary_df_tab34=binary_df_tab34,single_df_tab34=single_df_tab34,
                            binaryc_singles=binaryc_singles,beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,
                            gridpath=gridpath)[0], binary_df['m_init']))
            converted_elements = [convert_format(el) for el in isolist]
            inv_dict1 = create_inventory(dumping_arr, converted_elements, list_radio=list_radio)#todo: clarify what the dif is
            inv_dict_new1 = create_new_inventory(dumping_arr, converted_elements, list_radio=list_radio)
            dumping_arr,_ = tej_func(mean_timing_radio_list, dumping_arr, inv_dict1,inv_dict_new1,converted_elements=converted_elements)

    
    
    
    Yeff_all=[]
    for i,iso in enumerate(isolist):
        dumping_arr_iso=dumping_arr[np.where(isolist==iso)].flatten()
        
        Yeff_all.append(calc_Yeff_abs(iso,dM=dM,h=h,q_avg=q_avg,dumping_arr_iso=dumping_arr_iso,
                        beta_winds=beta_winds,beta_rlof=beta_rlof,beta_cc=beta_cc,gridpath=gridpath))
    print('elapsed = ' + str(time.time()-tinit))
    
    return Yeff_all

def calc_sep_weightings(bound_arr,nature_tag='loguniform'):
    bound_arr=np.sort(bound_arr)#just in case; you should probably do this for it anyway.
    #calculate the weightings (will add to 1) for different separations.
    weighting_arr=np.zeros(len(bound_arr)-1)
    
    
    log_bounds=np.log10(bound_arr)
    sep_weights=np.diff(log_bounds)/(log_bounds[-1]-log_bounds[0])
    
    return sep_weights
    
    
        
def write_preamble(f,h,beta_winds,beta_rlof,beta_cc,pretty_iso_list,date=''):
#     f.write('H Dummy preamble\n')
    f.write(f'H Effective binary yields for: h={h} beta_winds={beta_winds}, beta_rlof={beta_rlof}, and beta_cc={beta_cc}\n')
    f.write('H Underlying yields: Farmer et al., 2023\n')
    f.write(f'H Data prepared by: Alex Kemp, Tejpreet Kaur, {date}\n')
    f.write('H Isotopes:')
    for iso in pretty_iso_list:
        f.write(f' {iso},')
    f.write('\n')
    
    
def write_star_header(f,Z=0.0142,Minit=5,lifetime=5e9,Mfinal=2):
    
    lifetimestr= '%.3e' % Decimal(lifetime)
    Mfinalstr='%.4f' % Decimal(Mfinal)
    Minitstr='%.1f' % Decimal(Minit)
    f.write('H Table: (M=' + Minitstr + ',Z=' + str(Z)+')'+'\n')
    f.write('H Lifetime: ' +lifetimestr+'\n')
    f.write('H Mfinal: ' +Mfinalstr+'\n')
    f.write('&Isotopes &Yields     &X0         &Z &A'+'\n')
    
    
def write_star_data(f,iso,yieldnum,init_abu,Z,A,savename='dummy.txt'):
    #im pretty sure I'm eventually going to regret using Z for both metallicity and proton number...
    isostring='&'+pretty_iso(iso).ljust(9, ' ')
    Zstr='&'+str(Z).ljust(3, ' ')
    Astr='&'+str(A).ljust(3, ' ')
#     print(isostring + str('&'+'%.3e' % Decimal(yieldnum)).ljust(12, ' ') + str('&' + '%.3e' % Decimal(init_abu)).ljust(12) + Zstr + Astr+'\n')
    f.write(isostring + str('&'+'%.3e' % Decimal(yieldnum)).ljust(12, ' ') + str('&' + '%.3e' % Decimal(init_abu)).ljust(12) + Zstr + Astr+'\n')
    
def calc_init_abu_dict(gridpath=gridpath):
    isolist=make_iso_list_rob(gridpath)
    
    #initiallise:
    binary_df,single_df=read_iso_rob(isolist[0],gridpath)
    
    len_binary_df=len(binary_df)
    init_abu_dict={}
    init_abu_list=[]
    for i,iso in enumerate(isolist):
#         print(iso)
        binary_df,single_df=read_iso_rob(iso,gridpath)

        #check length:
        if len(binary_df)==len_binary_df:
            pass
        else:
            print(iso)
            print('df size missmatch! I assume same size binary_df for each isotope!')
            raise a
        valarr=np.array(binary_df['winds_init'] + binary_df['rlof_init'] + binary_df['cc_init'])
        init_abu_dict[iso]=valarr
        init_abu_list.append(valarr)
    init_abu_arr=np.array(init_abu_list)
    return init_abu_dict,init_abu_arr

def calc_out_abu_dict(gridpath=gridpath):
    isolist=make_iso_list_rob(gridpath)
    
    #initiallise:
    binary_df,single_df=read_iso_rob(isolist[0],gridpath)
    
    len_binary_df=len(binary_df)
    out_abu_dict={}
    out_abu_list=[]
    for i,iso in enumerate(isolist):
#         print(iso)
        binary_df,single_df=read_iso_rob(iso,gridpath)

        #check length:
        if len(binary_df)==len_binary_df:
            pass
        else:
            print(iso)
            print('df size missmatch! I assume same size binary_df for each isotope!')
            raise a
        valarr=np.array(binary_df['winds'] + binary_df['rlof'] + binary_df['cc'])
        out_abu_dict[iso]=valarr
        out_abu_list.append(valarr)
    out_abu_arr=np.array(out_abu_list)
    return out_abu_dict,out_abu_arr

def calc_net_abu_dict(gridpath=gridpath):
    isolist=make_iso_list_rob(gridpath)
    
    #initiallise:
    binary_df,single_df=read_iso_rob(isolist[0],gridpath)
    
    len_binary_df=len(binary_df)
    net_abu_dict={}
    net_abu_list=[]
    for i,iso in enumerate(isolist):
#         print(iso)
        binary_df,single_df=read_iso_rob(iso,gridpath)

        #check length:
        if len(binary_df)==len_binary_df:
            pass
        else:
            print(iso)
            print('df size missmatch! I assume same size binary_df for each isotope!')
            print(len(binary_df))
            print(len_binary_df)
            print(len(binary_df)==len_binary_df)
            raise a
        valarr=np.array(binary_df['winds']-binary_df['winds_init'] + binary_df['rlof']-binary_df['rlof_init'] + binary_df['cc']-binary_df['cc_init'])
        net_abu_dict[iso]=valarr
        net_abu_list.append(valarr)
    net_abu_arr=np.array(net_abu_list)
    return net_abu_dict,net_abu_arr