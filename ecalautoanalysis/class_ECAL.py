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
from typing import *


""" Global variables """

save_folder_global = 'Statistics' # Processed data from will be stored in a folder named like this. 
raw_data_folder_global = '/eos/home-s/spigazzi/Lab21/data/Reco/' # Raw data is stored here
plot_save_folder_global = 'Plots' # Produced plots are saved here


def gaussian(x: float=None, *p: tuple) -> float:
    """
    Returns a gaussian function with amplitude A, mean mu and std deviation sigma evaluated at x
    
    :param x: point at which the function is evaluated
    :param p: parameters of the gaussian; amplitude, mean, std deviation
    """
    A, mu, sigma = p
    return A * np.exp(-(x -mu)**2/(2*sigma**2))

""" Parent Class definition """

class ECAL:
    """
    Parent class of Amplitude, Time and Amplitude_Delta
    
    :param included_runs: run numbers to be analysed, eg. [15610, 15611]
    :param letters: corresponding to the boards connected, eg. ['A', 'B', 'D']
    :param save_folder: folder where the computed data should be stored
    :param raw_data_folder: folder where the raw experiment data is located
    :param plot_save_folder: folder where the plots are saved
    """
    
    def __init__(self, included_runs: List[int]=None, letters: List[str]=None, 
                 save_folder: str=save_folder_global, raw_data_folder: str=raw_data_folder_global, 
                 plot_save_folder: str=plot_save_folder_global):
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
        
        try: # checks the consistency of the boards and runs
            self.__check_consistency()
        except AssertionError as e:
            print(e)
        except TypeError as e:
            print(e)
   
            
    def __check_consistency(self):
        """
        Checks if the boards included in all the included_runs are the same, and checks if these boards are consistent with self.channel_names
        Also checks if included_runs is a list
        """
        
        # Check if included_runs is a list
        if type(self.included_runs) != list:
            raise TypeError("included_runs must be a list")
        
        # define the channels of the first run as channels_ref
        single_run = self.included_runs[0]
        folder =  self.raw_data_folder + str(int(single_run))
        h = uproot.concatenate({folder+'/*.root' : 'digi'}, allow_missing = True)
        columns_ref = ak.to_pandas(h).columns
        channels_ref = [channel for channel in columns_ref if channel[0] in ['A', 'B', 'C', 'D', 'E'] and channel[1] in self.numbers]
        
        # If inconsistency with self.channel_names, raise error
        if set(channels_ref) != set(self.channel_names):
            raise AssertionError("Letters do not match data")
        
        # Find the channels all the runs and check consistency with channels_ref
        for single_run in self.included_runs:
            
            folder =  self.raw_data_folder + str(int(single_run))
            h = uproot.concatenate({folder+'/*.root' : 'digi'}, allow_missing = True)

            columns = ak.to_pandas(h).columns
            
            # If inconsistency, raise error
            if set(columns) != set(columns_ref):
                raise AssertionError("Included runs are not consistent")
            