""" Imports """

import uproot
import numpy as np
import pandas as pd
import glob
import os
import h5py
import awkward as ak
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from decimal import *
from pathlib import Path


""" Global variables """

save_folder_global = 'Statistics' # Processed data from will be stored in a folder named like this. 
raw_data_folder_global = '/eos/home-s/spigazzi/Lab21/data/Reco/' # Raw data is stored here
plot_save_folder_global = 'Plots' # Produced plots are saved here


def gaussian(x, *p):
    """
    Returns a gaussian function with amplitude A, mean mu and std deviation sigma evaluated at x
    
    x --- (float): point at which the function is evaluated
    p --- (tuple of int): parameters of the gaussian; amplitude, mean, std deviation
    """
    A, mu, sigma = p
    return A * np.exp(-(x -mu)**2/(2*sigma**2))

""" Parent Class definition """

# TODO: check consistency of the different runs (same boards), raise error if not consistent
class ECAL:
    """
    parent class of Amplitude, Time and Amplitude_Delta
    
    included_runs --- (list of int) run numbers to be analysed, eg. [15610, 15611]
    letters --- (list of strings) corresponding to the boards connected, eg. ['A', 'B', 'D']
    save_folder --- (string) folder where the computed data should be stored
    raw_data_folder --- (string) folder where the raw experiment data is located
    plot_save_folder --- (string) folder where the plots are saved
    """
    
    def __init__(self, included_runs, letters, 
                 save_folder = save_folder_global, raw_data_folder = raw_data_folder_global, 
                 plot_save_folder = plot_save_folder_global):
        self.save_folder = save_folder
        self.raw_data_folder = raw_data_folder
        self.plot_save_folder = plot_save_folder

        self.numbers = ['1', '2', '3', '4', '5'] # The five channels on each board
        self.included_runs = included_runs
        self.letters = letters
        
        # for colormesh plots
        self.X = self.numbers.copy(); self.X.insert(0, '0')
        self.Y = self.letters.copy(); self.Y.insert(0, '0')

        # define channel_names, the access to the 'mesh' with the letters and the numbers
        self.channel_names = []
        for letter in self.letters:
            channel_names_temp = [letter + n for n in self.numbers]
            self.channel_names += channel_names_temp
        
        try:
            self.__check_consistency()
        except AssertionError as e:
            print(e)
   
            
    def __check_consistency(self):
        # TODO: docstring
        
        single_run = self.included_runs[0]
        folder =  self.raw_data_folder + str(int(single_run))
        h = uproot.concatenate({folder+'/*.root' : 'digi'}, allow_missing = True)

        columns_ref = ak.to_pandas(h).columns
        
        channels_ref = [channel for channel in columns_ref if channel[0] in ['A', 'B', 'C', 'D', 'E'] and channel[1] in self.numbers]
        
        if set(channels_ref) != set(self.channel_names):
            raise AssertionError("Letters do not match data")
        
        for single_run in self.included_runs:
            
            folder =  self.raw_data_folder + str(int(single_run))
            h = uproot.concatenate({folder+'/*.root' : 'digi'}, allow_missing = True)

            columns = ak.to_pandas(h).columns
            
            if set(columns) != set(columns_ref):
                raise AssertionError("Included runs are not consistent")
            