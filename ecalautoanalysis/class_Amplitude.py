""" Imports """

from .class_ECAL import *

""" 1st Child Class definition """

class Amplitude(ECAL):
    """
    This class is for the analysis of the amplitudes (not the resolution, cf. class Amplitude_Delta for this purpose).
    
    :param included_runs: list of all the numbers (int) corresponding to the runs to be analyzed, eg. [15610, 15611]
    :param letters: list of all the boards (str) connected for the included_runs, eg. ['A', 'B', 'D']
    :param save_folder: local path to the folder where files will be saved
    :param raw_data_folder: local path to the folder where the data from DQM is sent
    :param plot_save_folder: local path to the folder where the plots can be saved
    """
    def __init__(self, included_runs: List[int]=None, letters: List[str]=None,
                 save_folder: str=save_folder_global, raw_data_folder: str=raw_data_folder_global,
                 plot_save_folder: str=plot_save_folder_global):
        super().__init__(included_runs, letters, save_folder, raw_data_folder, plot_save_folder)
        
    # ------------------------------------------------------------------------------------------------------------------------------
    # GENERAL
    
    def __generate_stats(self, single_run: int=None, board: str=None, variation: str='run', plot: bool=False):
        """
        Generates the statistics for a given board in a run, either when analyzing spills or runs. Can also plot the histogram of the data.
        Statistics of the amplitude (mean, mean error, sigma, sigma error) are then saved in .csv files for later use.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        :param variation: ('run' or 'spill') computing the statistics per run or spill
        :param plot: boolean. If True, the histogram of the data is plotted.
        """
        folder =  self.raw_data_folder + str(int(single_run))

        # Computation with merged data: retrieve the amplitude
        folder =  self.raw_data_folder + str(int(single_run))
        h2 = uproot.concatenate({folder + '/*.root' : 'digi'}, allow_missing = True)
        run_name = os.path.basename(os.path.normpath(folder)) # creating folder to save csv file
        # TODO: delete print or add verbose boolean parameter?
        print('Run: ', run_name)
        run_save = self.save_folder + '/Run ' + run_name + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True) # folder created
        
        # retrieve only the channels for the given board
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        amp = h2['amp_max'] # retrieve the amplitude in the .root file
        amp_pd = pd.DataFrame(amp, columns=self.channel_names)[slicing]
        
        # column header
        col_list = len(self.numbers)*[board]; col_list = [x + y for x,y in zip(col_list, self.numbers)] 
        
        if variation=='spill': # if we want to compute the statistics per spill
            # retrieve the spill number in the .root file
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
                    hist, bin_edges = np.histogram(aspill_pd_temp[channel], bins = 1500)

                    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  

                    # fitting process: give a good guess to ECAL.__gaussian(*p)
                    mean_guess = np.average(bin_centers, weights=hist)
                    sigma_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist))
                    guess = [np.max(hist), mean_guess, sigma_guess]
                    
                    # fit the histogram with a gaussian, get the statistics
                    coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=5000)
                    mu = coeff[1]
                    mu_error = np.sqrt(covar[1,1])
                    sigma = coeff[2]
                    sigma_error = np.sqrt(covar[2,2])
                    
                    #store the statistics
                    mu_arr[i] = mu
                    mu_error_arr[i] = mu_error
                    sigma_arr[i] = sigma
                    sigma_error_arr[i] = sigma_error
                    
                    if plot: # TODO: add path name
                        title = f'Run: {run_name}, Channel: {board+self.numbers[i]}, Spill {spill}'
                        xlabel = 'Amplitude (??)'
                        ylabel = 'Occurence (a.u.)'
                        super()._ECAL__plot_hist(amp_pd, channel, bin_centers, title, xlabel, ylabel, *coeff)
                
                # gather all the statistics for each spill
                amp_mean_spill[j,:] = mu_arr
                amp_mean_err_spill[j,:] = mu_error_arr
                amp_sigma_spill[j,:] = sigma_arr
                amp_sigma_err_spill[j,:] = sigma_error_arr
                
            # convert the matrices to DataFrames
            spill_amp_mean_df = pd.DataFrame(amp_mean_spill, columns=col_list)
            spill_amp_mean_err_df = pd.DataFrame(amp_mean_err_spill, columns=col_list)
            spill_amp_sigma_df = pd.DataFrame(amp_sigma_spill, columns=col_list)
            spill_amp_sigma_err_df = pd.DataFrame(amp_sigma_err_spill, columns=col_list)
        
            # save these in .csv files: 4 files created per tuple (run, board)
            spill_amp_mean_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                     + f'/Spill mean amplitude run {single_run} board {board}.csv')
            spill_amp_mean_err_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                         + f'/Spill error mean amplitude run {single_run} board {board}.csv')
            spill_amp_sigma_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                      + f'/Spill sigma amplitude run {single_run} board {board}.csv')
            spill_amp_sigma_err_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                          + f'/Spill error sigma amplitude run {single_run} board {board}.csv')
        
        else: # if variation=='run'
            # empty arrays to store the statistics of each channel
            mu_arr = np.zeros(len(self.numbers))
            mu_error_arr = np.zeros(len(self.numbers))
            sigma_arr = np.zeros(len(self.numbers))
            sigma_error_arr = np.zeros(len(self.numbers))

            for i, channel in enumerate(slicing):
                hist, bin_edges = np.histogram(amp_pd[channel], bins = 1500)
                #hist, bin_edges, _ = plt.hist(amp_pd[channel], bins = 1500)

                bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  

                # fitting process: give a good guess
                mean_guess = np.average(bin_centers, weights=hist) # alternatively: mean_guess = bin_centers[np.argmax(hist)]
                sigma_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist))
                guess = [np.max(hist), mean_guess, sigma_guess]
                
                # fit the histogram with a gaussian
                coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=5000)
                
                # get the statistics from the fit, store them in the arrays
                mu = coeff[1]
                mu_error = np.sqrt(covar[1,1])
                sigma = coeff[2]
                sigma_error = np.sqrt(covar[2,2])
                mu_arr[i] = mu
                mu_error_arr[i] = mu_error
                sigma_arr[i] = sigma
                sigma_error_arr[i] = sigma_error
                
                if plot:
                    title = f'Run: {run_name}, Channel: {board+self.numbers[i]}'
                    xlabel = 'Amplitude (??)'
                    ylabel = 'Occurence (a.u.)'
                    
                    plot_save = self.plot_save_folder + f'/Run {single_run}'
                    Path(plot_save).mkdir(parents=True, exist_ok=True) # folder created
                    
                    path = plot_save + f'/Run amplitude run {single_run} board {board}'
                    
                    super()._ECAL__plot_hist(amp_pd, channel, bin_centers, title, xlabel, ylabel, path, *coeff)

            # convert the arrays into a single Dataframe
            run_amp_df = pd.DataFrame({'mu':mu_arr, 'mu error':mu_error_arr, 'sigma': sigma_arr, 'sigma error': sigma_error_arr})

            # save it in a single .csv file per tuple (run, board)
            run_amp_df.to_csv(self.save_folder + f'/Run {single_run}' 
                              + f'/Run amplitude run {single_run} board {board}.csv')
    
    
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
                                    + f'/Spill mean amplitude run {single_run} board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill error mean amplitude run {single_run} board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill sigma amplitude run {single_run} board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill error sigma amplitude run {single_run} board {board}.csv'))
            else: # if variation=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}'
                                   + f'/Run amplitude run {single_run} board {board}.csv')
                
        except FileNotFoundError:
            print('File not found, generating .csv')
            self.__generate_stats(single_run, board, variation) # generating the statistics file
            
            # loading the file and returning it
            if variation=='spill':
                return (pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill mean amplitude run {single_run} board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill error mean amplitude run {single_run} board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill sigma amplitude run {single_run} board {board}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}'
                                + f'/Spill error sigma amplitude run {single_run} board {board}.csv'))
            else: # if variation=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}'
                                   + f'/Run amplitude run {single_run} board {board}.csv')
            
        except: 
            raise Exception('Could not load nor generate .csv file')
            
            
    # ------------------------------------------------------------------------------------------------------------------------------
    # SPILLS
    
    def __amplitude_spill_single_board(self, single_run: int=None, board: str=None): 
        """
        Plots the amplitude per spill for a single board of a given single_run
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        """
        # TODO: update docstring
        
        # load the DataFrames with the statistics computed per spill
        mean, mean_err, sigma, sigma_err = self.__load_stats(single_run, board, 'spill')
        num_spills = mean.shape[0] # number of spills in the single run
        
        # keep only the channels for the board. Ex, if 'A', the ['A1', 'A2', etc.]
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        
        # Spill column in pd.DataFrame for plot
        spill_column_tmp = [len(self.numbers)*[i] for i in range(1, num_spills+1)]
        spill_column = []
        for lst in spill_column_tmp:
            spill_column += lst
        
        # Channel column in plot pd.DataFrame
        channel_column = num_spills*slicing
        
        # Mean and sigma columns in plot pd.DataFrame
        mean_arr = mean[slicing].to_numpy()
        mean_stacked = mean_arr.flatten()
        sigma_arr = sigma[slicing].to_numpy()
        sigma_stacked = sigma_arr.flatten()
        
        plot_df = pd.DataFrame({"spill": spill_column, "channel": channel_column, "mean": mean_stacked, "sigma": sigma_stacked})
      
        xlabel = 'Spill'
        ylabel = 'Amplitude (??)'
        plot_title = f'Run {single_run}, board {board}, mean amplitude over spills'
        
        super()._ECAL__plot_variation(plot_df, 'spill', xlabel, ylabel, plot_title)

    
    
    def __amplitude_spill_single_run(self, single_run: int=None):
        """
        Plots the amplitude per spill for a single_run (loops on its boards)
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        """
        for board in self.letters:
            self.__amplitude_spill_single_board(single_run, board)
    
    
    def variation_amplitude_spill(self):
        """
        Plots the amplitude per spill for all the runs in self.included_runs (loops on all the single_runs)
        """
        for single_run in self.included_runs:
            self.__amplitude_spill_single_run(single_run)

            
    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS
    
    # ---- HISTOGRAMS ----
    
    def __hist_amplitude_single_board(self, single_run: int=None, board: str=None):
        """
        Generates the statistics for all the channels on a given board. Plots the corresponding histograms.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        """
        self.__generate_stats(single_run, board, 'run', plot=True)
        

    def __hist_amplitude_single_run(self, single_run: int=None):
        """
        Generates the statistics for all the channels in a given run (loops on all its boards). Plots the corresponding histograms.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        """
        for board in self.letters:
            self.__hist_amplitude_single_board(single_run, board)
        

    def hist_amplitude(self):
        """
        Computes the statistics and plots the corresponding histogram for every single_run in self.included_runs
        """
        for single_run in self.included_runs:
            self.__hist_amplitude_single_run(single_run)
    
    
    # ---- VARIATION OVER RUNS ----
    
    def __amplitude_run_single_board(self, board: str=None):
        """
        Plots the mean amplitude over each single_run of self.included_runs for every channel in a given board
        
        :param board: board to be analyzed with the run, eg. 'C'
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
        
        plot_df = pd.DataFrame({"run": run_column, "channel": channel_column, "mean": mean_stacked, "sigma": sigma_stacked})
        
        display(plot_df)
        
        xlabel = 'Run'
        ylabel = 'Amplitude (??)'
        plot_title = f'Run {single_run}, board {board}, mean amplitude over runs'
        
        super()._ECAL__plot_variation(plot_df, 'run', xlabel, ylabel, plot_title)
    

    def variation_amplitude_run(self):
        """
        Plots the evolution of the mean amplitude over every single_run in self.included_runs.
        Warning: included_runs must be at least of length two.
        """
        try:
            if len(self.included_runs)  <= 1:
                raise ValueError('Need at least two runs to plot a variation')
            else:    
                for board in self.letters:
                    self.__amplitude_run_single_board(board)
        except ValueError as e:
            print(e)
    

    # ---- STATISTICS OVER RUNS ----
    
    def __run_statistics_single_run(self, single_run: int=None):
        """ 
        Plots the colormesh map with the mean amplitude (mu) over self.channel_names for a given single_run.
        Could also do the same with mu error, sigma, sigma error.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        """
        stat_names = ['Mu', 'Mu error', 'Sigma', 'Sigma_error']
        folder =  self.raw_data_folder + str(int(single_run))
        run_name = os.path.basename(os.path.normpath(folder))
        print('Run: ', run_name)
        run_save = self.save_folder + '/Run ' + str(run_name) + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        # TODO: do we also want to plot sigma, mu_err, sigma_err? if yes, then change docstring 
        mean = np.zeros((len(self.numbers), len(self.letters)))
        for i, board in enumerate(self.letters):
            run_amp_df = self.__load_stats(single_run, board, 'run')
            mean[:,i] = np.array(list(reversed(run_amp_df["mu"])))

        super()._ECAL__plot_colormesh(mean)
        
        
    def run_statistics(self):
        """
        Plots the colormesh map with the mean amplitude over self.channel_names for every single_run in self.included_runs.
        """
        for single_run in self.included_runs:
            self.__run_statistics_single_run(single_run)
