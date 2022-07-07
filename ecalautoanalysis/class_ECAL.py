""" Imports """

import uproot
import numpy as np
import pandas as pd
import os
import awkward as ak
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from decimal import *
from pathlib import Path
from typing import *

import plotly.io as pio

""" Global variables """

save_folder_global = 'Statistics' # Processed data from will be stored in a folder named like this. 
raw_data_folder_global = '/eos/home-s/spigazzi/Lab21/data/Reco/' # Raw data is stored here
plot_save_folder_global = 'Plots' # Produced plots will be saved here


def gaussian(x: float=None, *p: tuple) -> float:
    """
    Returns a gaussian function with amplitude A, mean mu and std deviation sigma evaluated at x
    
    :param x: point at which the function is evaluated
    :param p: parameters of the gaussian; amplitude, mean, std deviation
    :return: gaussian evaluated at the point x
    """
    A, mu, sigma = p
    return A * np.exp(-(x -mu)**2/(2*sigma**2))

""" Parent Class definition """

class ECAL:
    """
    This class is the parent class of Amplitude, Time_Delta and Amplitude_Delta. It contains the attributes and methods 
    that are to be inherited to the entire code structure. This class should be understood as 'virtual', in the sense that
    it is not possible to have an instance of ECAL. The boolean checked allows to skip the checking of the consistency of the runs
    if one already knows that they are consistent.
    
    :param included_runs: run numbers to be analysed, eg. [15610, 15611]
    :param letters: corresponding to the boards connected, eg. ['A', 'B', 'D']
    :param save_folder: folder where the computed data should be stored
    :param raw_data_folder: folder where the raw experiment data is located
    :param plot_save_folder: folder where the plots are saved
    :param checked: boolean stating whether included runs have already been checked or not
    """
    def __init__(self, included_runs: List[int]=None, letters: List[str]=None, 
                 save_folder: str=save_folder_global, raw_data_folder: str=raw_data_folder_global, 
                 plot_save_folder: str=plot_save_folder_global, checked: bool=False):
        self.save_folder = save_folder
        self.raw_data_folder = raw_data_folder
        self.plot_save_folder = plot_save_folder

        self.numbers = ['1', '2', '3', '4', '5'] # The five channels on each board
        self.included_runs = included_runs
        self.included_runs.sort() # Sorting the run names
        self.letters = letters
        self.clock_period = 6.238  # nanoseconds
        #self.n_bins = 0 # placeholder for n_bins in child classes

        # define channel_names, the access to the 'mesh' with the letters and the numbers
        self.channel_names = []
        for letter in self.letters:
            channel_names_temp = [letter + n for n in self.numbers]
            self.channel_names += channel_names_temp
        
        try: # checks the consistency of the boards and runs
            if not checked:
                self.__check_consistency()
        except AssertionError as e:
            print(e)
        except TypeError as e:
            print(e)
   

    # ------------------------------------------------------------------------------------------------------------------------------
    # GENERAL
    
    def __check_consistency(self):
        """
        Checks if the boards included in all the included_runs are the same, and checks if these boards are consistent with
        self.channel_names. Also checks if included_runs is indeed a list
        """
        print("Checking consistency of the included runs")

        # Check if included_runs is a list
        if type(self.included_runs) != list:
            raise TypeError("included_runs must be a list")
        
        # define the channels of the first run as channels_ref
        single_run = self.included_runs[0]
        folder =  self.raw_data_folder + str(int(single_run))
        try: 
            h = uproot.concatenate({folder+'/*.root' : 'digi'}, allow_missing = True)
        except FileNotFoundError as e: # if the file does not exist, we raise an exception
            print(e)
        else:
            columns_ref = ak.to_pandas(h).columns
            channels_ref = [channel for channel in columns_ref if channel[0] in ['A', 'B', 'C', 'D', 'E'] and channel[1] in self.numbers]

            # If inconsistency with self.channel_names, raise error
            if set(channels_ref) != set(self.channel_names):
                raise AssertionError("Letters do not match data")

            # Find the channels all the runs and check consistency with channels_ref
            for single_run in self.included_runs:
                print(f"Checking run {single_run}")
                folder =  self.raw_data_folder + str(int(single_run))
                try:
                    h = uproot.concatenate({folder+'/*.root' : 'digi'}, allow_missing = True)
                except FileNotFoundError as e: # if the file does not exist, we raise an exception
                    print(e)
                else:
                    columns = ak.to_pandas(h).columns

                    # If inconsistency, raise error
                    if set(columns) != set(columns_ref):
                        raise AssertionError("Included runs are not consistent")


    def __three_gaussians(self, x: float=None, *p: tuple) -> float:
        """
        Returns a sum of three gaussians with the same std deviation, and with means separated by one clock period, with parameters given by *p, evaluated at the point x
        
        :param x: point at which the function is evaluated
        :param p: parameters of the gaussians; [amplitude1, mean1, std deviation1, amplitude2, amplitude3]
        
        :return: sum of three gaussians evaluated at the point x, with period self.clock_period
        """

        A1, mu1, sigma1, A2, A3 = p # get the coefficients
        coeff1 = (A1, mu1, sigma1)
        coeff2 = (A2, mu1+self.clock_period*1000, sigma1)
        coeff3 = (A3, mu1-self.clock_period*1000, sigma1)
        # the gaussians have the same std deviations
        return gaussian(x, *coeff1) + gaussian(x, *coeff2) + gaussian(x, *coeff3) 
       
        
    def __plot_hist(self, df: pd.DataFrame=None, channel: str=None, bin_centers: np.array=None, 
                    hist_title: str=None, xlabel: str=None, ylabel: str=None, path: str=None, file_title: str=None, class_type: str=None, *coeff: tuple):
        """
        Plots the histogram of the DataFrame df for a single channel and with the bin_centers given. Title and labels are 
        also included in the arguments, as well as the path to save the figure and a tuple with the coefficients for the 
        (multiple) gaussian(s) fit of the data. The name of the saved file is file_title.
        
        :param df: DataFrame containing the data to be plotted
        :param channel: the channel we want to study
        :param bin_centers: placement of the bars of the histogram
        :param hist_title: title of the figure
        :param xlabel: label of the x-axis
        :param ylabel: label of the y-axis
        :param path: path to save the figure
        :param file_title: title of the file (figure) saved
        :param class_type: either 'amplitude', 'time_delta' or 'amplitude_delta'
        :param *coeff: pointer to the coefficients computed with the (multiple) gaussian(s) fit
        """
        # Plotting the data
        #trace1 = px.histogram(df, x=channel, nbins=2*self.n_bins)
        trace1 = px.bar(df, x='bin_centers', y='hist')
        bin_width = (np.max(bin_centers) - np.min(bin_centers)) / len(bin_centers)
        trace1.update_traces(width=bin_width)
        trace1.update_traces(marker=dict(line=dict(width=0)))
        #trace1.update_traces(width=bin_width, marker_line_width=0, selector=dict(type="bar"))
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(trace1.data[0]) # plot the DataFrame    
        #fig.update_layout(bargap=0, bargroupgap = 0)
        
        # Compute reduced chi2
        if len(coeff) == 3:
            r = df["hist"].to_numpy() - gaussian(bin_centers, *coeff)
            dof = len(df["hist"].to_numpy()) - 3 # Number of degrees of freedom = nb data points - nb parameters
        else:
            r = df["hist"].to_numpy() - self.__three_gaussians(bin_centers, *coeff)
            dof = len(df["hist"].to_numpy()) - 5 # Number of degrees of freedom = nb data points - nb parameters

        yerror = np.sqrt(df["hist"].to_numpy())
        chisq = np.sum([(r[i]/yerror[i])**2 for i in range(len(r)) if yerror[i] != 0]) / dof # Reduced chi squared
        # Get correct units
        if class_type == 'amplitude' or class_type == 'amplitude_delta':
            unit = 'ADC counts'
        else:
            unit = 'ps'
        
        if len(coeff) == 3: # if we only have a gaussian
            d = {'x': bin_centers, 'y': gaussian(bin_centers, *coeff)}

            amp, mean, sigma = coeff
            fig.add_vline(x=mean, line_dash='dash', line_color='red')
            fig.add_vrect(x0=mean-sigma, x1=mean+sigma, line_width=0, fillcolor='red', opacity=0.2)
            fig.add_annotation(text=f'Mean: {round(mean,2)} {unit},<br>std dev: {round(sigma,2)} {unit},<br>reduced chi squared: {round(chisq,0)}', xref='x domain', yref='y domain', x=0.9, y=0.8, showarrow=False)
        
        else: # if we have more than 3 parameters in coeff, then it means that we work with three gaussians
            d = {'x': bin_centers, 'y': self.__three_gaussians(bin_centers, *coeff)}
        
        # Ploting the fit
        fit_pd = pd.DataFrame(data=d)
        trace2 = px.line(fit_pd, x='x', y='y', color_discrete_sequence=['red'])
        fig.add_trace(trace2.data[0], secondary_y=False) # plot the fit

        fig.update_layout(title={'text': hist_title, 'y':0.98, 'x':0.5, 'xanchor': 'center'},
                         xaxis_title=xlabel,
                         yaxis_title=ylabel,
                         font = dict(size=18),
                         margin=dict(l=30, r=20, t=50, b=20))
        
        # Save the figures
        pio.full_figure_for_development(fig, warn=False)
        fig.write_image(path + file_title + '.png')
        fig.write_image(path + file_title + '.pdf')
        fig.write_image(path + file_title +'.svg')
        fig.write_html(path + file_title + '.html')

        
    def __plot_variation(self, df: pd.DataFrame=None, variation: str=None,
                         xlabel: str=None, ylabel: str=None, plot_title: str=None, path: str=None, file_title: str=None, class_type: str=None, df_ratio: pd.DataFrame=None):
        """
        Plots the variation either over runs or spills of the DataFrame. Title and labels of the axes are included 
        as arguments, as well as the path to the saving folder and the title of the file.
        
        :param df: DataFrame containing the data to be plotted
        :param variation: either 'run' (histograms are computed over a full run) or 'spill' (separately for each spill in single_run).
        :param xlabel: label of the x-axis
        :param ylabel: label of the y-axis
        :param plot_title: title of the figure
        :param path: path to the folder where the plot is saved
        :param file_title: title of the file (figure) saved
        :param class_type: either 'amplitude', 'time_delta' or 'amplitude_delta'
        :param df_ratio: dataframe with the ratio of the amplitudes used with variation='run' and class_type='amplitude'
        """
        # In the case of run amplitude variation, we also want to plot the gain to see at which point it switches from 1 to 10
        if variation == 'run' and class_type == 'amplitude':
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            trace1 = px.line(data_frame=df, x=variation, y='mean', error_y="sigma", color='channel')
            for trace_data in trace1.data:
                fig.add_trace(trace_data)                       
            trace2 = px.line(data_frame=df, x=variation, y='gain', color='channel', line_dash_sequence=['dash'])    
            for trace_data in trace2.data:
                fig.add_trace(trace_data, secondary_y=True)

            xlabel = 'Laser power (au)'
            fig.add_hline(y=4000, line_width=2, line_dash="dash", line_color="black")
            fig.update_yaxes(title_text="Gain", secondary_y=True)


            # ratio plot
            fig2 = make_subplots(specs=[[{"secondary_y": False}]])
            trace1bis = px.line(data_frame=df_ratio, x=variation, y='ratio', error_y="error", color='channel')
            for trace_data in trace1bis.data:
                fig2.add_trace(trace_data)                       

            fig2.update_yaxes(title_text="Amplitude ratio to first channel", secondary_y=False) 


        else: # variation = 'spill' or class_type not 'amplitude'
            fig = px.line(data_frame=df, x=variation, y='mean', error_y="sigma", color='channel')

        fig.update_layout(title={'text': plot_title, 'y':0.98, 'x':0.5, 'xanchor': 'center'},
                         xaxis_title=xlabel,
                         yaxis_title=ylabel,
                         font = dict(size=18),
                         margin=dict(l=30, r=20, t=50, b=20))

        # Change ticks of x-axis depending on variation
        if variation == 'spill':
            fig.update_layout(xaxis= dict(tickmode='linear', tick0=1, dtick=1))
        else:
            #tick_list = [str(run) for run in self.included_runs]
            tick_list = list(np.arange(140, 232, 2)) # Input power instead of run number

            if len(self.included_runs) > 10: # if too many runs, do not show xtick for each run
                for i in range(len(tick_list)):
                    if i%4 != 0:
                        tick_list[i] = ''
       
            fig.update_layout(xaxis= dict(tickmode='array', tickvals=np.arange(len(self.included_runs)), ticktext=tick_list))
            # Also set xticks for ratio plot
            if class_type == 'amplitude':
                fig2.update_layout(xaxis= dict(tickmode='array', tickvals=np.arange(len(self.included_runs)), ticktext=tick_list))
            
        # Save the figures
        pio.full_figure_for_development(fig, warn=False)
        fig.write_image(path + file_title + '.png')
        fig.write_image(path + file_title + '.pdf')
        fig.write_image(path + file_title +'.svg')
        fig.write_html(path + file_title + '.html')  
    
        # If run variation of amplitude, also plot the ratio of the amplitudes with respect to channel 1 of the board
        if variation == 'run' and class_type == 'amplitude':
            pio.full_figure_for_development(fig2, warn=False)
            fig2.write_image(path + file_title + ' ratio.png')
            fig2.write_image(path + file_title + ' ratio.pdf')
            fig2.write_image(path + file_title +' ratio.svg')
            fig2.write_html(path + file_title + ' ratio.html')

    def __plot_colormesh(self, mean: np.array=None, plot_title: str=None, path: str=None, file_title: str=None, class_type: str=None):
        """
        Plots a 2D colormesh map of the mean of a given quantity (amplitude, amplitude difference, time difference) over all channels
        and boards. 
        
        :param mean: array containing all the data
        :param plot_title: title of the figure
        :param path: path to the folder where the plot is saved
        :param file_title: title of the file (figure) saved
        :param class_type: either 'amplitude', 'time_delta' or 'amplitude_delta'
        """
        # Formatting DataFrame for the plot
        mean_df = pd.DataFrame(mean)
        mean_df.columns = self.letters
        indices = list(reversed(self.numbers))
        mean_df.index = indices
        
        if class_type == 'amplitude':
            unit = 'ADC counts'
        else:
            unit = 'ps'        

        # Plotting the colormesh
        fig = px.imshow(mean_df,
                        labels=dict(x="Board", y="Channel", color=unit),
                        x=self.letters,
                        y=indices
                       )

        fig.update_layout(title={'text': plot_title, 'y':0.98, 'x':0.5, 'xanchor': 'center'},
                         font = dict(size=18),
                         margin=dict(l=30, r=20, t=50, b=20))
        
        # Save the figures
        pio.full_figure_for_development(fig, warn=False)
        fig.write_image(path + file_title + '.png')
        fig.write_image(path + file_title + '.pdf')
        fig.write_image(path + file_title + '.svg')
        fig.write_html(path + file_title + '.html')

        
