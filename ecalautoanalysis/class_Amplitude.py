""" Imports """

from .class_ECAL import *

"""
General function definitions
"""

def sigma_amp_fit(A: float=None, *p: tuple) -> float:
    """
    Fit function for the relative amplitude resolution
    
    :param A: amplitude
    :param p: parameters N (noise), s (stochastic), and c (offset)
    
    :return: Value of the function at A with parameters N, s, and c
    """
    N, s, c = p
    return np.sqrt( (N/A)**2 + (s**2/A) + c**2 )

"""
Child class definition
"""

class Amplitude(ECAL):
    """
    This class serves for the analysis of the amplitude measurements and amplitude resolution of the detector. With a given list of self.included_runs, one can plot amplitude histograms, variation of the amplitude over runs, colormeshes over the channels, as well as the relative amplitude resolution using the public methods. Has the number of bins used in the histograms as an attribute.
    
    :param included_runs: list with all the numbers (int) corresponding to the runs to be analyzed, eg. [15610, 15611]
    :param letters: list with all the boards (str) connected for the included_runs, eg. ['A', 'B', 'D']
    :param save_folder: local path to the folder where files will be saved
    :param raw_data_folder: local path to the folder where the data from DQM is sent
    :param plot_save_folder: local path to the folder where the plots can be saved
    :param n_bins: number of bins in the ampitude histograms
    :param checked: if checked==True, assumes that the consistency of self.included_runs has already been checked, so it does not apply the consistency check
    """
    def __init__(self, included_runs: List[int]=None, letters: List[str]=None,
                 save_folder: str=save_folder_global, raw_data_folder: str=raw_data_folder_global,
                 plot_save_folder: str=plot_save_folder_global, checked: bool=False):
        super().__init__(included_runs, letters, save_folder, raw_data_folder, plot_save_folder, checked)
        self.n_bins = 50   
        
        
    # ------------------------------------------------------------------------------------------------------------------------------
    # GENERAL
    
    def __generate_stats(self, single_run: int=None, board: str=None, variation: str='run', plot: bool=False, spill_index: int=None):
        """
        Generates the statistics for a given board in a run, either when analyzing spills or runs. Can also plot the histogram of the data. Statistics of the amplitude Gaussian fit (mean, mean error, sigma, sigma error) are then saved in .csv files for later use. Also saves the gain for the analysis of the linearity of the system.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        :param variation: ('run' or 'spill') computing the statistics per run or spill, 'run' per default.
        :param plot: boolean. If True, the histogram of the data is plotted. False per default.
        :param spill_index: Index of the spill to consider when variation='spill'
        """
        try:
            if board not in self.letters:
                raise ValueError("Board must be included in the list of letters")
                
            folder =  self.raw_data_folder + str(int(single_run))

            # Computation with merged data: retrieve the amplitude
            folder =  self.raw_data_folder + str(int(single_run))
            if variation == 'spill' and plot:
                try:
                    h2 = uproot.concatenate({folder + f'/{spill_index}.root': 'digi'}, allow_missing=True)
                    if(type(h2)==list):
                        raise TypeError
                except FileNotFoundError as e:
                    print(e)
                    return -1
                except TypeError as e:
                    print(f"Spill {spill_index} in run {single_run} is either empty or incomplete, skipping this spill.")
                    return -1
            else:
                h2 = uproot.concatenate({folder + '/*.root' : 'digi'}, allow_missing = True)

            run_name = os.path.basename(os.path.normpath(folder)) # creating folder to save csv file
            print('Run: ', run_name)
            run_save = self.save_folder + '/Run ' + run_name + '/'
            Path(run_save).mkdir(parents=True, exist_ok=True) # folder created

            # retrieve only the channels for the given board
            slicing = [channel for channel in self.channel_names if channel[0] == board]
            amp = h2['amp_max'] # retrieve the amplitude in the .root file
            amp_pd = pd.DataFrame(amp, columns=self.channel_names)[slicing]
               
            # remove nonsensical values from root file
            for channel in slicing:
                amp_pd = amp_pd[(amp_pd[channel] < 40000)] # max ADC counts possible is 40000

            # Get gain
            gain = h2['gain']
            gain_pd = pd.DataFrame(gain, columns=self.channel_names)[slicing]

            # column header
            col_list = len(self.numbers)*[board]; col_list = [x + y for x,y in zip(col_list, self.numbers)] 

            if variation=='spill': # if we want to compute the statistics per spill
                # retrieve the spill number in the .root file
                if plot:
                    h1 = uproot.concatenate({folder + f'/{spill_index}.root' : 'h4'}, allow_missing = True)
                else:
                    h1 = uproot.concatenate({folder + '/*.root' : 'h4'}, allow_missing = True)
                spill = h1['spill'] 
                spill_pd = pd.DataFrame(spill, columns=["spill_nb"])

                # merge the two DataFrames
                aspill_pd = pd.concat([amp_pd, spill_pd], axis=1, join='inner')
                
                # create empty matrices to store the statistics
                spill_set = set(aspill_pd["spill_nb"]) # to get a set of unique spill numbers
                amp_mean_spill = np.zeros((len(spill_set), len(self.numbers)))
                amp_mean_err_spill = np.zeros((len(spill_set), len(self.numbers)))
                amp_sigma_spill = np.zeros((len(spill_set), len(self.numbers)))
                amp_sigma_err_spill = np.zeros((len(spill_set), len(self.numbers)))

                for j, spill in enumerate(spill_set):
                    aspill_pd_temp = aspill_pd[aspill_pd.spill_nb == spill]

                    # 'empty' arrays to store the statistics of each channel
                    mu_arr = np.zeros(len(self.numbers))
                    mu_error_arr = np.zeros(len(self.numbers))
                    sigma_arr = np.zeros(len(self.numbers))
                    sigma_error_arr = np.zeros(len(self.numbers))

                    for i, channel in enumerate(slicing):         
                        hist, bin_edges = np.histogram(aspill_pd_temp[channel], bins = self.n_bins)

                        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  

                        # fitting process: give a good guess to ECAL.__gaussian(*p)
                        mean_guess = np.average(bin_centers, weights=hist)
                        sigma_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist))
                        guess = [np.max(hist), mean_guess, sigma_guess]

                        # fit the histogram with a gaussian, get the statistics
                        try:
                            coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=10000)
                        except RuntimeError as e:
                            print(e)
                            print(f"Fit unsuccessful, arbitrary coefficients set to {guess} and covariance matrix to 0.")                   
                            coeff = guess
                            covar = np.zeros((3,3))  
                        
                        mu = coeff[1]
                        mu_error = np.sqrt(covar[1,1])
                        sigma = coeff[2]
                        sigma_error = np.sqrt(covar[2,2])

                        #store the statistics
                        mu_arr[i] = mu
                        mu_error_arr[i] = mu_error
                        sigma_arr[i] = sigma
                        sigma_error_arr[i] = sigma_error

                        if plot:
                            df = pd.DataFrame({'bin_centers': bin_centers, 'hist': hist})

                            title = f'Run: {run_name}, channel: {board+self.numbers[i]}, spill {spill}'
                            xlabel = 'Amplitude (ADC counts)'
                            ylabel = 'Occurence (a.u.)'
                            
                            file_title = f'Amplitude channel {board+self.numbers[i]} spill {spill}'
                            plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/histogram/'
                            Path(plot_save).mkdir(parents=True, exist_ok=True)
                            super()._ECAL__plot_hist(df, channel, bin_centers, title, xlabel, ylabel, plot_save, file_title, 'amplitude', *coeff)

                    # gather all the statistics for each spill
                    amp_mean_spill[j,:] = mu_arr
                    amp_mean_err_spill[j,:] = mu_error_arr
                    amp_sigma_spill[j,:] = sigma_arr
                    amp_sigma_err_spill[j,:] = sigma_error_arr

                if not plot:
                    # convert the matrices to DataFrames
                    spill_amp_mean_df = pd.DataFrame(amp_mean_spill, columns=col_list)
                    spill_amp_mean_err_df = pd.DataFrame(amp_mean_err_spill, columns=col_list)
                    spill_amp_sigma_df = pd.DataFrame(amp_sigma_spill, columns=col_list)
                    spill_amp_sigma_err_df = pd.DataFrame(amp_sigma_err_spill, columns=col_list)

                    # Spill list for spill variation
                    spill_single_df = pd.DataFrame({'spills': list(spill_set)})
                    spill_single_df.to_csv(self.save_folder + f'/Run {single_run}' + f'/Spill spill list amplitude board {board}.csv')

                    # save these in .csv files: 4 files created per tuple (run, board)
                    spill_amp_mean_df.to_csv(self.save_folder + f'/Run {single_run}'
                                             + f'/Spill mean amplitude board {board}.csv')
                    spill_amp_mean_err_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                                 + f'/Spill error mean amplitude board {board}.csv')
                    spill_amp_sigma_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                              + f'/Spill sigma amplitude board {board}.csv')
                    spill_amp_sigma_err_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                                  + f'/Spill error sigma amplitude board {board}.csv')

            else: # if variation=='run'
                # empty arrays to store the statistics of each channel
                mu_arr = np.zeros(len(self.numbers))
                mu_error_arr = np.zeros(len(self.numbers))
                sigma_arr = np.zeros(len(self.numbers))
                sigma_error_arr = np.zeros(len(self.numbers))

                for i, channel in enumerate(slicing):
                    hist, bin_edges = np.histogram(amp_pd[channel], bins = self.n_bins)

                    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  

                    # fitting process: give a good first guess
                    mean_guess = np.maximum(0, np.average(bin_centers, weights=hist)) # mean amplitude should be positive
                    sigma_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist))
                    guess = [np.max(hist), mean_guess, sigma_guess]
                    
                    print("channel", channel)
                    
                    # fit the histogram with a gaussian
                    try:
                        bound = ([0, 0, 0], [np.inf, 30000, 30000])
                        coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=10000, bounds=bound)
                    except RuntimeError as e:
                        print(e)
                        print(f"Fit unsuccessful, arbitrary coefficients set to {guess} and covariance matrix to 0.")                   
                        coeff = guess
                        covar = np.zeros((3,3))
                    except ValueError as e:
                        print(e)
                        print(f"Fit with given guess unsuccessful, arbitrary coefficients set to {guess} and covariance matrix to 0.")                   
                        coeff = guess
                        covar = np.zeros((3,3))                  

                    # get the statistics from the fit, store them in the arrays
                    mu = coeff[1]
                    mu_error = np.sqrt(covar[1,1])
                    sigma = coeff[2]
                    sigma_error = np.sqrt(covar[2,2])
                    mu_arr[i] = mu
                    mu_error_arr[i] = mu_error
                    sigma_arr[i] = sigma
                    sigma_error_arr[i] = sigma_error

                    if plot: # make the plot via ECAL
                        df = pd.DataFrame({'bin_centers': bin_centers, 'hist': hist})

                        title = f'Run: {run_name}, Channel: {board+self.numbers[i]}'
                        xlabel = 'Amplitude (ADC counts)'
                        ylabel = 'Occurence (a.u.)'

                        plot_save = self.plot_save_folder + f'/Run {single_run}'
                        Path(plot_save).mkdir(parents=True, exist_ok=True) # folder created

                        file_title = f'Amplitude channel {board+self.numbers[i]}'
                        plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/histogram/'
                        Path(plot_save).mkdir(parents=True, exist_ok=True)

                        super()._ECAL__plot_hist(df, channel, bin_centers, title, xlabel, ylabel, plot_save, file_title, 'amplitude', *coeff)

                # convert the arrays into a single Dataframe
                run_amp_df = pd.DataFrame({'mu':mu_arr, 'mu error':mu_error_arr, 'sigma': sigma_arr, 'sigma error': sigma_error_arr})

                # save it in a single .csv file per tuple (run, board)
                run_amp_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                  + f'/Run amplitude run {single_run} board {board}.csv')

                # Save gain in csv file
                gain_df = pd.DataFrame()
                for channel in slicing: 
                    # select the gain
                    mask10 = gain_pd[channel] == 10
                    mask1 = gain_pd[channel] == 1 
                    # count the number of times a given gain appears
                    n_10 = mask10.sum()
                    n_1 = mask1.sum()
                    # keep the gain that is the most present in the run for each channel
                    if n_10 >= n_1:
                        gain_df[channel] = [10]
                    else:
                        gain_df[channel] = [1]
                # save in .csv
                gain_df.to_csv(self.save_folder + f'/Run {single_run}' + f'/Run gain board {board}.csv')

        except ValueError as e:
            print(e)
    
    
    def __load_stats(self, single_run: int=None, board: str=None, variation: bool=None) -> Union[tuple, pd.DataFrame]:
        """
        Loads the file containing the statistics for a single couple (run, board). If the file does not exist, calls __generate_stats()
        Returns the .csv file(s) of __generate_file()
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        :param variation: ('run' or 'spill'). if __generate is called, we compute the statistics per run or spill
        
        :return: pd.DataFrame (one if variation='run', four if variation='spill') with the statistics
        """
        try: # check if the file exists
            if variation=='spill':
                return (pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                    + f'/Spill mean amplitude board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill error mean amplitude board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill sigma amplitude board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill error sigma amplitude board {board}.csv'))
            else: # if variation=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}'
                                   + f'/Run amplitude run {single_run} board {board}.csv')
                
        except FileNotFoundError:
            print('File not found, generating .csv')
            self.__generate_stats(single_run, board, variation) # generating the statistics file
            # loading the file and returning it
            if variation=='spill':
                return (pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill mean amplitude board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill error mean amplitude board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill sigma amplitude board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill error sigma amplitude board {board}.csv'))
            else: # if variation=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}'
                                   + f'/Run amplitude run {single_run} board {board}.csv')
            
        except: 
            raise Exception('Could not load nor generate .csv file')
       
    def get_mean(self, single_run: int=None, board: str=None) -> pd.core.series.Series:
        """
        Getter method for the mean of the amplitude Gaussian fit for the channels in the board in thelution('test code 280622') single_run. Returns a container with the mean amplitude for each of the channels in the board.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        
        :return: pandas.core.series.Series containing the mean of the amplitude for each of the channels in the board
        """
        df = self.__load_stats(single_run, board, variation='run')
        return df["mu"]
    
    
    def get_sigma(self, single_run: int=None, board: str=None) -> pd.core.series.Series:
        """
        Getter method for the standard deviation of the amplitude Gaussian fit for the channels in the board in the single_run. Returns a container with the standard deviation amplitude for each of the channels in the board.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        
        :return: pandas.core.series.Series containing the standard deviation of the amplitude for each of the channels in the board
        """
        df = self.__load_stats(single_run, board, variation='run')
        return df["sigma"]
    
    
    def get_mean_err(self, single_run: int=None, board: str=None) -> pd.core.series.Series:
        """
        Getter method for the error on the mean of the amplitude Gaussian fit for the channels in the board in the single_run. Returns a container with the error on the mean amplitude for each of the channels in the board.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        
        :return: pandas.core.series.Series containing the error on the mean of the amplitude for each of the channels in the board
        """
        df = self.__load_stats(single_run, board, variation='run')
        return df["mu error"]
    
    
    def get_sigma_err(self, single_run: int=None, board: str=None) -> pd.core.series.Series:
        """
        Getter method for the error on the standard deviation of the amplitude Gaussian fit for the channels in the board in the single_run. Returns a container with the error on the standard deviation amplitude for each of the channels in the board.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        
        :return: pandas.core.series.Series containing the error on the standard deviation of the amplitude for each of the channels in the board
        """
        df = self.__load_stats(single_run, board, variation='run')
        return df["sigma error"]
        
            
    # ------------------------------------------------------------------------------------------------------------------------------
    # SPILLS
    
    def __spill_single_board(self, single_run: int=None, board: str=None): 
        """
        Plots the mean amplitude as a function of spill number for a single board of a given single_run.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        """
        # load the DataFrames with the statistics computed per spill
        mean, mean_err, sigma, sigma_err = self.__load_stats(single_run, board, 'spill')
        num_spills = mean.shape[0] # number of spills in the single run
        
        # keep only the channels for the board. Ex, if 'A', the ['A1', 'A2', etc.]
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        
        # Retrieve spill list in csv file (already generated through load_stats)
        spill_df = pd.read_csv( self.save_folder + f'/Run {single_run}' + f'/Spill spill list amplitude board {board}.csv' )        
        spill_lst = list(spill_df["spills"])

        # Spill column in pd.DataFrame for plot
        spill_column_tmp = [len(self.numbers)*[i] for i in spill_lst]
        spill_column = []
        for lst in spill_column_tmp:
            spill_column += lst
        
        # Channel column in plot pd.DataFrame
        channel_column = num_spills*slicing
        
        # Mean and std dev columns in plot pd.DataFrame
        mean_arr = mean[slicing].to_numpy()
        mean_stacked = mean_arr.flatten()
        sigma_arr = sigma[slicing].to_numpy()
        sigma_stacked = sigma_arr.flatten()
        
        plot_df = pd.DataFrame({"spill": spill_column, "channel": channel_column, "mean": mean_stacked, "sigma": sigma_stacked})
      
        xlabel = 'Spill'
        ylabel = 'Amplitude (ADC counts)'
        plot_title = f'Run {single_run}, board {board}, mean amplitude over spills'
        
        file_title = f'Amplitude board {board}'
        plot_save = self.plot_save_folder + '/Run ' + str(single_run) + '/variation_spill/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        super()._ECAL__plot_variation(plot_df, 'spill', xlabel, ylabel, plot_title, plot_save, file_title)

    
    def __spill_single_run(self, single_run: int=None):
        """
        Plots the amplitude per spill for a single_run (loops on its boards)
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        """
        for board in self.letters:
            self.__spill_single_board(single_run, board)
    
    
    def spill_variation(self):
        """
        Plots the amplitude per spill for all the runs in self.included_runs (loops on all the single_runs)
        """
        for single_run in self.included_runs:
            self.__spill_single_run(single_run)

            
    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS

    # ---- HISTOGRAMS ----
    
    def __hist_single_board(self, single_run: int=None, board: str=None, variation: str=None, spill_i: int=None):
        """
        Generates the statistics for all the channels on a given board and plots the corresponding histograms. If variation is "run", then the histograms contain all the events in the single_run. If variation is "spill", the histogram contains the events in the spill_i of the single_run.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        :param variation: either "run" or "spill". If variation is "run", then the histograms contain all the events in the single_run. If variation is "spill", the histogram contains the events in the spill_i of the single_run.
        :param spill_i: index of the spill to be considered in the case variation="spill"
        """
        self.__generate_stats(single_run, board, variation, plot=True, spill_index=spill_i)
        

    def __hist_single_run(self, single_run: int=None, variation: str=None, spill_i: int=None):
        """
        Generates the statistics for all the channels in a given run (loops on all its boards) and plots the corresponding histograms.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param variation: either "run" or "spill". If variation is "run", then the histograms contain all the events in the single_run. If variation is "spill", the histogram contains the events in the spill_i of the single_run.
        :param spill_i: index of the spill to be considered in the case variation="spill"
        """
        for board in self.letters:
            self.__hist_single_board(single_run, board, variation, spill_i)
        

    def hist(self, variation: str='run', spill_i: int=None):
        """
        Computes the statistics and plots the corresponding histograms for every single_run in self.included_runs and channels in self.channel_names.
        
        :param variation: either "run" or "spill". If variation is "run", then the histograms contain all the events in the run considered. If variation is "spill", the histogram contains the events in the spill_i of the run considered.
        :param spill_i: index of the spill to be considered in the case variation="spill"
        """
        for single_run in self.included_runs:
            self.__hist_single_run(single_run, variation, spill_i)
    
    
    # ---- VARIATION OVER RUNS ----
    
    def __run_single_board(self, board: str=None, file_title: str=None):
        """
        Plots the mean amplitude over each single_run of self.included_runs for every channel in a given board
        
        :param board: board to be analyzed with the run, eg. 'C'
        :param file_title: name of the figure files to be saved
        """
        # empty matrices to store the statistics     
        mean = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma = np.zeros((len(self.included_runs), len(self.numbers)))
        # load the DataFrames and fill the matrices
        for i, single_run in enumerate(self.included_runs):
            run_amp_df = self.__load_stats(single_run, board, 'run') # 4 columns, n_numbers rows
            mean[i,:] = run_amp_df["mu"]
            sigma[i,:] = run_amp_df["sigma"]
        
        # keep only the channels of the board we are interested in
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        
        # Run column in pd.DataFrame for plot
        run_column_tmp = [len(self.numbers)*[i] for i in np.arange(len(self.included_runs))]
        run_column = []
        for lst in run_column_tmp:
            run_column += lst
        
        # Channel column in plot pd.DataFrame
        channel_column = len(self.included_runs)*slicing
        
        # Mean and sigma columns in plot pd.DataFrame
        mean_stacked = mean.flatten()
        sigma_stacked = sigma.flatten()
        
        # Get gain from csv (already generated by load_stats above)
        gain_column = []
        for single_run in self.included_runs:
            gain_df = pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Run gain board {board}.csv')
            gain_column += list(gain_df.iloc[0])[1:]
        
        # make dataframe for the plot
        plot_df = pd.DataFrame({"run": run_column, "channel": channel_column, "mean": mean_stacked, "sigma": sigma_stacked, 'gain': gain_column})
        
        # also make a dataframe with the ratio to the first channel of the board
        mean_ratio = mean / mean[:,0].reshape(len(mean[:,0]), 1)
        sigma_ratio = mean_ratio * ( np.abs(sigma[:,0] / mean[:,0]).reshape(len(sigma[:,0]), 1) + np.abs(sigma / mean))
        mean_ratio = mean_ratio.flatten()
        sigma_ratio = sigma_ratio.flatten()
        plot_ratio = pd.DataFrame({"run": run_column, "channel": channel_column, "ratio": mean_ratio, "error": sigma_ratio})
        
        # labels and title of the figure
        xlabel = 'Run'
        ylabel = 'Amplitude (ADC counts)'
        plot_title = f'Board {board}, mean amplitude over runs'
        file_title = file_title + f' board {board}'

        # path and save folder
        plot_save = self.plot_save_folder + '/run_variation/amplitude/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        
        # make the plot via ECAL
        super()._ECAL__plot_variation(plot_df, 'run', xlabel, ylabel, plot_title, plot_save, file_title, 'amplitude', plot_ratio)
    

    def run_variation(self, file_title: str=None):
        """
        Plots the evolution of the mean amplitude over every single_run in self.included_runs.
        Warning: included_runs must be at least of length two.
        
        :param file_title: name of the figure files to be saved
        """
        try: # Check if enough included_runs to plot a variation
            if len(self.included_runs)  <= 1:
                raise ValueError('Need at least two runs to plot a variation')
            else:    
                for board in self.letters:
                    self.__run_single_board(board, file_title)
        except ValueError as e:
            print(e)
    

    # ---- STATISTICS OVER RUNS ----
    
    def __run_colormesh_single_run(self, single_run: int=None):
        """ 
        Plots the colormesh map with the mean amplitude (mu) over self.channel_names for a given single_run. Could also be used to plot the colormesh map with of the sigma (+ associated uncertainties)
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        """
        #stat_names = ['Mu', 'Mu error', 'Sigma', 'Sigma_error']
        
        # Making directory for csv files
        folder =  self.raw_data_folder + str(int(single_run))
        run_name = os.path.basename(os.path.normpath(folder))
        print('Run: ', run_name)
        run_save = self.save_folder + '/Run ' + str(run_name) + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        # Loading mean from csv files or generating it if necessary
        mean = np.zeros((len(self.numbers), len(self.letters)))
        for i, board in enumerate(self.letters):
            run_amp_df = self.__load_stats(single_run, board, 'run')
            mean[:,i] = np.array(list(reversed(run_amp_df["mu"])))

        # Generating the plot
        plot_title = f'Run {single_run}, mean amplitudes'
        file_title = 'Mean Amplitude'
        plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/colormesh/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        # make the plot via ECAL
        super()._ECAL__plot_colormesh(mean, plot_title, plot_save, file_title, 'amplitude')
        
        
    def run_colormesh(self):
        """
        Plots the colormesh map with the mean amplitude over self.channel_names for every single_run in self.included_runs. Could also be used to plot the colormesh map of the sigma of the amplitude (+ associated uncertainties)
        """
        for single_run in self.included_runs:
            self.__run_colormesh_single_run(single_run)
            
            
    # RESOLUTION ----------------------------------------------------------------------------------

    def __resolution_single_board(self, board: str=None, file_title: str=None):
        """
        Plots for each channel in the board given the relative amplitude resolution as a function of the amplitude. Displays the fitted parameters.
        
        :param board: board considered
        :param file_title: name of the figure files to be saved
        """
        # Arrays with the stats for all included runs and channels in the board
        A_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        A_err_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma_err_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        
        # Filling the arrays
        for i, single_run in enumerate(self.included_runs):
            A_lst[i] = self.get_mean(single_run, board)
            sigma_lst[i] = self.get_sigma(single_run, board)
            A_err_lst[i] = self.get_mean_err(single_run, board)
            sigma_err_lst[i] = self.get_sigma_err(single_run, board)
            
        # Plot the resolution for all channels
        for j, channel in enumerate([board+number for number in self.numbers]):
            yerror = sigma_lst[:,j]/A_lst[:,j] * np.sqrt( (A_err_lst[:,j]/A_lst[:,j])**2 + (sigma_err_lst[:,j]/sigma_lst[:,j])**2 )
            guess = [30, 2, 0.02] # suitable guess (trial and error)

            # Mask for the plot because data doesn't fit the model for high amplitudes
            mask = (A_lst[:,j] > 0) & (A_lst[:,j] < 10000) 

            coeff, covar = curve_fit(sigma_amp_fit, A_lst[:,j][mask], sigma_lst[:,j][mask]/A_lst[:,j][mask], p0=guess, sigma=yerror[mask], maxfev=10000)
            
            print(f'channel {channel}')
            
            # create figure
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Plotting the data with errorbars
            df_data = pd.DataFrame({'x': A_lst[:,j], 'y': sigma_lst[:,j]/A_lst[:,j], "err_x": A_err_lst[:,j], "err_y": yerror})
            trace1 = px.scatter(data_frame=df_data, x='x', y='y', error_x="err_x", error_y="err_y", color_discrete_sequence=["Crimson"])
            fig.add_trace(trace1.data[0])
            
            # Plotting the fit
            x = np.linspace(np.min(A_lst[:,j]), np.max(A_lst[:,j]))
            df_fit = pd.DataFrame({'x': x, 'y': sigma_amp_fit(x, *coeff)})
            trace2 = px.line(df_fit, x='x', y='y')
            fig.add_trace(trace2.data[0], secondary_y=False)
            
            # Computing the chi2
            r = sigma_lst[:,j]/A_lst[:,j] - sigma_amp_fit(A_lst[:,j], *coeff)
            dof = len(sigma_lst[:,j]) - 3 # Number of degrees of freedom = nb data points - nb parameters
            chisq = np.sum((r/yerror)**2) / dof # Reduced chi squared

            # Printing the coefficients in the fit
            fig.add_annotation(text=f'Parameters: <br> N={round(coeff[0],2)} ADC counts, <br> s={round(coeff[1],2)} ADC^1/2, <br> c={round(coeff[2],2)}<br>Reduced chi squared: {round(chisq,0)}', xref='x domain', yref='y domain', x=0.4, y=0.8, xanchor='left', align='left', showarrow=False) 
            
            plot_title = f"Amplitude relative resolution, channel {channel}"
            xlabel = "Average amplitude (ADC count)"
            ylabel = "Relative amplitude resolution"
            fig.update_layout(xaxis=dict(title=xlabel),
                              yaxis=dict(title=ylabel),
                              title={'text': plot_title, 'y':0.98, 'x':0.5, 'xanchor': 'center'},
                              font = dict(size=18),
                              margin=dict(l=30, r=20, t=50, b=20))

            fig.update_layout(updatemenus=[ # button to change the scale of the axis to linear, semilogy or loglog
                                           dict(
                                               buttons = [
                                                           dict(label="Linear",
                                                                method="relayout",
                                                                args=[{"yaxis.type": "linear", "xaxis.type": "linear"}]),
                                                           dict(label="Semilog y",
                                                                method="relayout",
                                                                args=[{"yaxis.type": "log", "xaxis.type": "linear"}]),
                                                           dict(label="Loglog",
                                                                method="relayout",
                                                                args=[{"yaxis.type": "log", "xaxis.type": "log"}])
                                                         ]
                                               )
                                          ]
                             )

            # Saving the figures
            plot_save = self.plot_save_folder + '/resolution/amplitude/'
            Path(plot_save).mkdir(parents=True, exist_ok=True)
            
            path = plot_save
            pio.full_figure_for_development(fig, warn=False)
            fig.write_image(path + file_title + f' channel {channel}' + '.png')
            fig.write_image(path + file_title + f' channel {channel}' + '.pdf')
            fig.write_image(path + file_title + f' channel {channel}' + '.svg')
            fig.write_html(path + file_title + f' channel {channel}' + '.html')


    def resolution(self, file_title: str=None):
        """
        Plots for each channels in self.channel_names the relative amplitude resolution as a function of the amplitude.
        
        :param file_title: name of the figure files to be saved
        """
        try: # Checking if there are enough runs for the resolutin fit
            if len(self.included_runs)  <= 2:
                raise ValueError('Need at least three runs to fit the three parameters for the amplitude resolution')
            for board in self.letters:
                self.__resolution_single_board(board, file_title)
                
        except ValueError as e:
            print(e)


    def __resolution_all_single_board(self, board: str=None, file_title: str=None):
        """
        Plots the relative amplitude resolution as a function of the amplitude for each channel in the board given in argument, in a single plot with fits to the function sigma_amp_fit for each channel. Does not display the fitted parameters.
        
        :param board: board considered
        :param file_title: name of the figure files to be saved
        """
        # Arrays with the stats for all included runs and channels in the board
        A_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        A_err_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma_err_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        
        # Filling the arrays
        for i, single_run in enumerate(self.included_runs):
            A_lst[i] = self.get_mean(single_run, board)
            sigma_lst[i] = self.get_sigma(single_run, board)
            A_err_lst[i] = self.get_mean_err(single_run, board)
            sigma_err_lst[i] = self.get_sigma_err(single_run, board)
        
        # Dataframes to plot the data and the fits respectively
        plot_df = pd.DataFrame(columns=['x', 'y', 'err_x', 'err_y', 'channel'])
        fits_df = pd.DataFrame(columns=['x', 'y', 'channel'])

        # Plot the resolution for all channels
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        for j, channel in enumerate([board+number for number in self.numbers]):
            # Error from Gaussian error propagation
            yerror = sigma_lst[:,j]/A_lst[:,j] * np.sqrt( (A_err_lst[:,j]/A_lst[:,j])**2 + (sigma_err_lst[:,j]/sigma_lst[:,j])**2 )
            
            # Dataframe for a single channel
            df_data = pd.DataFrame({'x': A_lst[:,j], 'y': sigma_lst[:,j]/A_lst[:,j], "err_x": A_err_lst[:,j], "err_y": yerror})
            
            # Column containing the channel name for each of the entries of df_data
            channel_column = [channel] * len(self.included_runs)
            
            # Add column with the channel name
            df_data['channel'] = channel_column
            
            # Append at the bottom of plot_df
            plot_df = pd.concat([plot_df, df_data], axis=0)
            
            # Guess for the fit (trial and error)
            guess = [30, 2, 0.02]
            
            # Mask for the plot because data doesn't fit the model for high amplitudes
            mask = df_data["x"] > 0 & (df_data["x"] < 10000)
            coeff, covar = curve_fit(sigma_amp_fit, df_data["x"][mask], df_data["y"][mask], p0=guess, sigma=df_data["err_y"][mask], maxfev=10000)            
            
            # Plotting the fit (linspace to have almost continuous trend)
            xx = np.linspace(np.min(df_data["x"]), np.max(df_data["x"]), 100)
            df_fit = pd.DataFrame({'x': xx, 'y': sigma_amp_fit(xx, *coeff)})
            
            # Channel column for the fit dataframe
            channel_column_fit = [channel] * len(xx)
            df_fit['channel'] = channel_column_fit
            # Append at the bottom of fits_df
            fits_df = pd.concat([fits_df, df_fit], axis=0)
        
        # Plot all the data points
        trace1 = px.scatter(data_frame=plot_df, x='x', y='y', error_x="err_x", error_y="err_y", color='channel')
        for trace_data in trace1.data:
            fig.add_trace(trace_data)
        
        # Plot the fits
        trace2 = px.line(fits_df, x='x', y='y', color='channel')
        for trace_data in trace2.data:
            fig.add_trace(trace_data) # plot the fits

        plot_title = f"Amplitude relative resolution, board {board}"
        xlabel = "Average amplitude (ADC count)"
        ylabel = "Relative amplitude resolution"
        fig.update_layout(xaxis=dict(title=xlabel),
                          yaxis=dict(title=ylabel),
                          title={'text': plot_title, 'y':0.98, 'x':0.5, 'xanchor': 'center'},
                          font = dict(size=18),
                          margin=dict(l=30, r=20, t=50, b=20))

        fig.update_layout(updatemenus=[ # add the option to change the scale of the axis to linear, semilogy or loglog
                                       dict(
                                           buttons = [
                                                       dict(label="Linear",
                                                            method="relayout",
                                                            args=[{"yaxis.type": "linear", "xaxis.type": "linear"}]),
                                                       dict(label="Semilog y",
                                                            method="relayout",
                                                            args=[{"yaxis.type": "log", "xaxis.type": "linear"}]),
                                                       dict(label="Loglog",
                                                            method="relayout",
                                                            args=[{"yaxis.type": "log", "xaxis.type": "log"}])
                                                     ]
                                           )
                                      ]
                         )

        # Saving the figures
        plot_save = self.plot_save_folder + '/resolution/amplitude/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        
        path = plot_save
        pio.full_figure_for_development(fig, warn=False)
        fig.write_image(path + file_title + f' board {board}' + '.png')
        fig.write_image(path + file_title + f' board {board}' + '.pdf')
        fig.write_image(path + file_title + f' board {board}' + '.svg')
        fig.write_html(path + file_title + f' board {board}' + '.html')


    def resolution_all(self, file_title: str=None):
        """
        Plots for each board in self.letters the relative amplitude resolution as a function of the amplitude with a fit to the function sigma_amp_fit.
        
        :param file_title: name of the figure files to be saved
        """
        try: # Checking if there are enough runs for the resolutin fit
            if len(self.included_runs)  <= 2:
                raise ValueError('Need at least three runs to fit the three parameters for the amplitude resolution')
                
            for board in self.letters:
                self.__resolution_all_single_board(board, file_title)
        except ValueError as e:
            print(e)
