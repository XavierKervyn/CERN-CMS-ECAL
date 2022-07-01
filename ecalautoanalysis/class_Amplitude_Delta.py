""" Imports """

from .class_ECAL import *

""" Child Class definition """     
        
class Amplitude_Delta(ECAL):
    """
    This class is for the analysis of the amplitude difference between the channels of the detector.
    
    With a given list of self.included_runs and a reference_channel, one can plot amplitude delta histograms, variation of the amplitude delta over runs, colormeshes over the channels, as well as the relative amplitude resolution using the public methods.
    
    :param included_runs: list of all the numbers corresponding to the runs to be analyzed
    :param letters: list of all the letters corresponding to the boards connected for the included_runs
    :param save_folder: local path to the folder where files will be saved
    :param raw_data_folder: local path to the folder where the data from DQM is sent
    :param plot_save_folder: local path to the folder where the plots can be saved
    """
    def __init__(self, included_runs: List[int]=None, letters: List[str]=None,
                 save_folder: str=save_folder_global, raw_data_folder: str=raw_data_folder_global,
                 plot_save_folder: str=plot_save_folder_global, checked: bool=False):
        super().__init__(included_runs, letters, save_folder, raw_data_folder, plot_save_folder, checked)
        
        # class attribute
        self.n_bins = 50
    
    
    # ------------------------------------------------------------------------------------------------------------------------------
    # GENERAL
    
    def __compute_delta(self, amp: pd.DataFrame=None, board: str=None, ref_channel: str=None) -> pd.DataFrame:
        """ 
        Computes the amplitude difference (delta) for all channels in a board, wrt. a given reference channel. 
        Also returns the mu and sigma statistics and their errors.
        
        :param amp: DataFrame with the amplitude of all the channels in self.channel_names
        :param board: letter corresponding to the board to be analyzed, eg. 'A'
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'

        :return: pd.DataFrame with the differences in amplitude: amplitude - amplitude_ref 
        """
        amp_pd = amp
        n_numbers = len(self.numbers)

        amp_delta_pd = pd.DataFrame()

        slicing = [channel for channel in self.channel_names if channel[0]==board]
        for i, channel in enumerate(slicing):
            if channel == ref_channel:
                continue

            reference_amp = amp_pd[ref_channel]
            curr_amp = amp_pd[channel]
            amp_delta = curr_amp - reference_amp

            # Save amplitude deltas for later analysis
            amp_delta_pd[f'{channel}'] = amp_delta
            
        return amp_delta_pd
    
    
    def __generate_stats(self, single_run: int=None, board: str=None, ref_channel: str=None, 
                         variation: str='run', plot: bool=False, spill_index: int=None):
        """ 
        Generates the statistics for a given board in a run, either when analyzing spills or runs. Can also plot the histogram of the data.
        Statistics of the amplitude delta Gaussian fit (mean, mean error, sigma, sigma error) are then saved in .csv files for later use.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param variation: ('run' or 'spill') computing the statistics per run or spill
        :param plot: boolean. If True, the histogram of the data is plotted.
        """
        try:
            if ref_channel not in self.channel_names:
                raise ValueError("Reference channel must be in the channel list")
                
            elif board not in self.letters:
                raise ValueError("Board must be included in the list of letters")
                
            else:
                # Computation with merged data
                folder =  self.raw_data_folder + str(int(single_run))
                if variation=='spill' and plot==True:
                    try: # raises an exception if the index references a non-existing spill
                        h2 = uproot.concatenate({folder + f'/{spill_index}.root' : 'digi'}, allow_missing = True)
                    except FileNotFoundError as e:
                        print(e)
                        return -1
                else:
                    h2 = uproot.concatenate({folder + '/*.root' : 'digi'}, allow_missing = True)

                run_name = os.path.basename(os.path.normpath(folder))
                print('Run: ', run_name)
                run_save = self.save_folder + '/Run ' + run_name + '/'
                Path(run_save).mkdir(parents=True, exist_ok=True)

                slicing = [channel for channel in self.channel_names if channel[0]==board]

                ref_idx = h2[ref_channel][0]
                amp = h2['amp_max']
                amp_pd = pd.DataFrame(amp, columns=self.channel_names)

                # column header for the Dataframes
                col_list = len(self.numbers)*[board]; col_list = [x + y for x,y in zip(col_list, self.numbers)]

                if variation=='spill':
                    # Computation with merged data: retrieve the spill number
                    if plot==True:
                        h1 = uproot.concatenate({folder + f'/{spill_index}.root' : 'h4'}, allow_missing = True)
                    else:
                        h1 = uproot.concatenate({folder + '/*.root' : 'h4'}, allow_missing = True)
                    spill = h1['spill'] 
                    spill_pd = pd.DataFrame(spill, columns=["spill_nb"]) 
                    
                    # merge the two Dataframes
                    aspill_pd = pd.concat([amp_pd, spill_pd], axis=1, join='inner')

                    # create empty arrays to store the statistics
                    spill_set = set(aspill_pd["spill_nb"]) # set of unique spill numbers
                    amp_mean_spill = np.zeros((len(spill_set), len(self.numbers)))
                    amp_mean_err_spill = np.zeros((len(spill_set), len(self.numbers)))
                    amp_sigma_spill = np.zeros((len(spill_set), len(self.numbers)))
                    amp_sigma_err_spill = np.zeros((len(spill_set), len(self.numbers)))

                    for j, spill in enumerate(spill_set):
                        aspill_pd_temp = aspill_pd[aspill_pd.spill_nb == spill]

                        amp_delta_pd = self.__compute_delta(aspill_pd_temp, board, ref_channel)

                        # 'empty' arrays to store the statistics of each channel
                        mu_arr = np.zeros(len(self.numbers))
                        mu_error_arr = np.zeros(len(self.numbers))
                        sigma_arr = np.zeros(len(self.numbers))
                        sigma_error_arr = np.zeros(len(self.numbers))

                        for i, channel in enumerate(slicing):
                            if channel == ref_channel:
                                continue

                            hist, bin_edges = np.histogram(amp_delta_pd[channel], bins = self.n_bins)

                            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)

                            # fitting process
                            mean_guess = np.average(bin_centers, weights=hist)
                            sigma_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist))

                            guess = [np.max(hist), mean_guess, sigma_guess]
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
                            mu_arr[i] = mu
                            mu_error_arr[i] = mu_error
                            sigma_arr[i] = sigma
                            sigma_error_arr[i] = sigma_error

                            if plot:
                                df = pd.DataFrame({'bin_centers': bin_centers, 'hist': hist})

                                title = f'Run: {run_name}, channel: {board+self.numbers[i]}, ref {ref_channel}, spill {spill}'
                                xlabel = 'Amplitude delta (ADC counts)'
                                ylabel = 'Occurence (a.u.)'
                                
                                file_title = f'Amplitude Delta channel {board+self.numbers[i]} ref {ref_channel} spill {spill}'
                                
                                plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/histogram/'
                                Path(plot_save).mkdir(parents=True, exist_ok=True)
                                super()._ECAL__plot_hist(df, channel, bin_centers, title, xlabel, ylabel, plot_save, file_title, 'amplitude', *coeff)

                        amp_mean_spill[j,:] = mu_arr
                        amp_mean_err_spill[j,:] = mu_error_arr
                        amp_sigma_spill[j,:] = sigma_arr
                        amp_sigma_err_spill[j,:] = sigma_error_arr

                    # convert the matrices to Dataframes
                    spill_amp_mean_df = pd.DataFrame(amp_mean_spill, columns=col_list)
                    spill_amp_mean_err_df = pd.DataFrame(amp_mean_err_spill, columns=col_list)
                    spill_amp_sigma_df = pd.DataFrame(amp_sigma_spill, columns=col_list)
                    spill_amp_sigma_err_df = pd.DataFrame(amp_sigma_err_spill, columns=col_list)

                    # save these in .csv files
                    spill_amp_mean_df.to_csv(self.save_folder 
                                             + f'/Run {single_run}' 
                                             + f'/Spill mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
                    spill_amp_mean_err_df.to_csv(self.save_folder 
                                                 + f'/Run {single_run}' 
                                                 + f'/Spill error mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
                    spill_amp_sigma_df.to_csv(self.save_folder 
                                              + f'/Run {single_run}' 
                                              + f'/Spill sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
                    spill_amp_sigma_err_df.to_csv(self.save_folder 
                                                  + f'/Run {single_run}' 
                                                  + f'/Spill error sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv')

                else: # if variation=='run':
                    amp_delta_pd = self.__compute_delta(amp_pd, board, ref_channel)

                    # 'empty' arrays to store the statistics of each channel
                    mu_arr = np.zeros(len(self.numbers))
                    mu_error_arr = np.zeros(len(self.numbers))
                    sigma_arr = np.zeros(len(self.numbers))
                    sigma_error_arr = np.zeros(len(self.numbers))

                    for i, channel in enumerate(slicing):
                        if channel == ref_channel:
                            continue

                        hist, bin_edges = np.histogram(amp_delta_pd[channel], bins = self.n_bins)

                        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  

                        # fitting process
                        mean_guess = np.average(bin_centers, weights=hist)
                        sigma_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist))

                        guess = [np.max(hist), mean_guess, sigma_guess]
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
                        mu_arr[i] = mu
                        mu_error_arr[i] = mu_error
                        sigma_arr[i] = sigma
                        sigma_error_arr[i] = sigma_error

                        if plot:
                            df = pd.DataFrame({'bin_centers': bin_centers, 'hist': hist})

                            title = f'Run: {run_name}, Channel: {board+self.numbers[i]}, Ref {ref_channel}'
                            xlabel = 'Amplitude delta (ADC counts)'
                            ylabel = 'Occurence (a.u.)'
                            
                            file_title = f'Amplitude Delta channel {board+self.numbers[i]} ref {ref_channel}'
                                
                            plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/histogram/'
                            Path(plot_save).mkdir(parents=True, exist_ok=True)
                            super()._ECAL__plot_hist(df, channel, bin_centers, title, xlabel, ylabel, plot_save, file_title, 'amplitude',*coeff)

                    # convert the arrays into a single Dataframe
                    run_amp_delta_df = pd.DataFrame({'mu':mu_arr, 'mu error':mu_error_arr,
                                                     'sigma': sigma_arr, 'sigma error': sigma_error_arr})

                    # save it in a .csv file
                    run_amp_delta_df.to_csv(self.save_folder 
                                            + f'/Run {single_run}' 
                                            + f'/Run amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
        except ValueError as e:
            print(e)

            
    def __load_stats(self, single_run: int=None, board: str=None, ref_channel: str=None, variation: bool=None) -> Union[tuple, pd.DataFrame]:
        """
        Loads the file containing the statistics for a single triplet (run, board, ref_channel). 
        If the file does not exist, calls __generate_stats().
        Returns the DataFrames generated from the .csv file(s).
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param variation: ('run' or 'spill'). if __generate is called, we compute the statistics per run or spill
        
        :return: pd.DataFrame (one if variation='run', four if variation='spill') with the statistics
        """
        try: # check if the file exists
            
            if variation=='spill': # returns a tuple with the 4 files
                return (pd.read_csv(self.save_folder 
                                    + f'/Run {single_run}' 
                                    + f'/Spill mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder 
                                + f'/Run {single_run}' 
                                + f'/Spill error mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder 
                                + f'/Run {single_run}' 
                                + f'/Spill sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder 
                                + f'/Run {single_run}' 
                                + f'/Spill error sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv'))
            else: # if variation=='run':
                return pd.read_csv(self.save_folder 
                                   + f'/Run {single_run}' 
                                   + f'/Run amplitude delta run {single_run} board {board} ref {ref_channel}.csv')   
            
        except FileNotFoundError:
   
            print('File not found, generating .csv')
            self.__generate_stats(single_run, board, ref_channel, variation) # generating the statistics file
        
            # loading the file and returning it
            if variation=='spill': # returns a tuple with the 4 files
                return (pd.read_csv(self.save_folder 
                                    + f'/Run {single_run}' 
                                    + f'/Spill mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder 
                                + f'/Run {single_run}' 
                                + f'/Spill error mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder 
                                + f'/Run {single_run}' 
                                + f'/Spill sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder 
                                + f'/Run {single_run}' 
                                + f'/Spill error sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv'))

            else: # if variation=='run':
                return pd.read_csv(self.save_folder 
                                   + f'/Run {single_run}' 
                                   + f'/Run amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
        except:
            raise Exception('Could not load nor generate .csv file')
            
            
    # ------------------------------------------------------------------------------------------------------------------------------
    # SPILLS
    
    def __spill_single_board(self, single_run: int=None, board: str=None, ref_channel: str=None): 
        """
        Plots the evolution of the amplitude delta per spill for a given board in a single run. 
        The delta is taken wrt. the reference channel ref_channel.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        """
        # load the Dataframes
        mean, mean_err, sigma, sigma_err = self.__load_stats(single_run, board, ref_channel, 'spill')
        num_spills = mean.shape[0] # number of spills in the single run
        
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
        ylabel = 'Amplitude delta (ADC counts)'
        plot_title = f'Run {single_run}, board {board}, ref {ref_channel}, mean amplitude delta over spills'
        
        file_title = f'Amplitude Delta board {board} ref {ref_channel}'
        plot_save = self.plot_save_folder + '/Run ' + str(single_run) + '/variation_spill/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        super()._ECAL__plot_variation(plot_df, 'spill', xlabel, ylabel, plot_title, plot_save, file_title)

        
    def __spill_single_run(self, single_run: int=None, ref_channel: str=None, all_channels: bool=None):
        """
        Plots the evolution of the amplitude delta per spill for a single run.
        The delta is taken wrt. the reference channel ref_channel.
        If all_channels is true, the plot is done for all other channels in self.channel_names.
        Otherwise, the plot is done for all channels in the board corresponding to the reference channel, eg. 'A' if ref_channel = 'A1'.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param all_channels: plotting either for all channels self.channel_names or only for those in the board to which ref_channel belongs
        """
        if all_channels:
            for board in self.letters:
                self.__spill_single_board(single_run, board, ref_channel)
        else:
            board = ref_channel[0]
            self.__spill_single_board(single_run, board, ref_channel)
    
    
    def spill_variation(self, ref_channel: str=None, all_channels: bool=None):
        """
        Plots the evolution of the amplitude delta per spill for all the single runs in self.included_runs.
        The delta is taken wrt. the reference channel ref_channel.
        If all_channels is true, the plot is done for all other channels in self.channel_names.
        Otherwise, the plot is done for all channels in the board corresponding to the reference channel, eg. 'A' if ref_channel = 'A1'.
        
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param all_channels: plotting either for all channels self.channel_names or only for those in the board to which ref_channel belongs
        """
        for single_run in self.included_runs:
            self.__spill_single_run(single_run, ref_channel, all_channels)

            
    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS
    
    # --- HISTOGRAMS ---
    
    def __hist_single_board(self, single_run: int=None, board: str=None, ref_channel: str=None, variation: str='run', spill_i: int=None):
        """ 
        Generates the statistics for a run and plots the histograms of the amplitude delta for the channels in the board considered.
        If variation is "run", then the histograms contain all the events in the single_run. If variation is "spill", the histogram contains the events in the spill_i of the single_run.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param board: board to be analyzed with the run, eg. 'C'
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        """
        self.__generate_stats(single_run, board, ref_channel, variation, plot=True, spill_index=spill_i)
        
            
    def __hist_single_run(self, single_run: int=None, ref_channel: str=None, all_channels: bool=None, variation: str='run', spill_i: int=None):
        """ 
        Generates the statistics for a run and plots the histograms of the amplitude delta for all the channels considered.
        Does it either for all channels self.channel_names or only for those in the board to which ref_channel belongs.
        If variation is "run", then the histograms contain all the events in the single_run. If variation is "spill", the histogram contains the events in the spill_i of the single_run.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param all_channels: plotting either for all channels self.channel_names or only for those in the board to which ref_channel belongs
        :param variation: either "run" or "spill". If variation is "run", then the histograms contain all the events in the single_run. If variation is "spill", the histogram contains the events in the spill_i of the single_run.
        :param spill_i: index of the spill to be considered in the case variation="spill"
        """
        if all_channels:
            for board in self.letters:
                self.__hist_single_board(single_run, board, ref_channel, variation, spill_i)
        else:
            board = ref_channel[0]
            self.__hist_single_board(single_run, board, ref_channel, variation, spill_i)
    
    
    def hist(self, ref_channel: str=None, all_channels: bool=None, variation: str='run', spill_i: int=None):
        """
        Generate the statistics and plots the histogram of the amplitude delta of all the channels considered for all the runs in self.included_runs. The delta is taken with respect to the ref_channel for every single run in self.included_runs.
        Does it either for all channels self.channel_names or only for those in the board to which ref_channel belongs.
        If variation is "run", then the histograms contain all the events in the single_run. If variation is "spill", the histogram contains the events in the spill_i of the single_run.

        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param all_channels: plotting either for all channels self.channel_names or only for those in the board to which ref_channel belongs
        :param variation: either "run" or "spill". If variation is "run", then the histograms contain all the events in the single_run. If variation is "spill", the histogram contains the events in the spill_i of the single_run.
        :param spill_i: index of the spill to be considered in the case variation="spill"
        """
        for single_run in self.included_runs:
            self.__hist_single_run(single_run, ref_channel, all_channels, variation, spill_i)

            
    # --- VARIATION OVER RUNS ---
    
    def __run_single_board(self, board: str=None, ref_channel: str=None, file_title: str=None):
        """
        Plots evolution of the mean amplitude over self.included_runs for given board and reference channel
        
        :param board: board to be analyzed with the run, eg. 'C'
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param file_title: name of the figure files to be saved
        """
        # load the Dataframes     
        mean = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma = np.zeros((len(self.included_runs), len(self.numbers)))
        for i, single_run in enumerate(self.included_runs):
            run_amplitude_delta_df = self.__load_stats(single_run, board, ref_channel, 'run') # 4 columns, n_numbers rows
            mean[i,:] = run_amplitude_delta_df["mu"]
            sigma[i,:] = run_amplitude_delta_df["sigma"] 
        
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
        
        xlabel = 'Run'
        ylabel = 'Amplitude delta (ADC counts)'
        plot_title = f'Board {board}, ref {ref_channel}, mean amplitude delta over runs'
        
        plot_save = self.plot_save_folder + '/run_variation/amplitude_delta/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        super()._ECAL__plot_variation(plot_df, 'run', xlabel, ylabel, plot_title, plot_save, file_title)
    

    def run_variation(self, ref_channel: str=None, all_channels: bool=None, file_title: str=None):
        """
        Plots evolution of the mean amplitude over self.included_runs for a given reference channel
        Does it either for all channels self.channel_names or only for those in the board to which ref_channel belongs
        
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param all_channels: plotting either for all channels self.channel_names or only for those in the board to which ref_channel belongs
        :param file_title: name of the figure files to be saved
        """
        try: # must have at least two runs included to plot a variation over runs
            if len(self.included_runs) <= 1:
                raise ValueError("Need at least two runs to plot a variation")
            else:
                if all_channels:
                    for board in self.letters:
                        self.__run_single_board(board, ref_channel, file_title)
                else:
                    board = ref_channel[0]
                    self.__run_single_board(board, ref_channel, file_title)
        except ValueError as e:
            print(e)
            
    
    # --- STATISTICS OVER RUNS ---
    
    def __run_colormesh_single_run(self, single_run: int=None, ref_channel: str=None, file_title: str=None):
        """ 
        Plots the colormesh map with the mean amplitude (mu) over self.channel_names for a given single_run.
        
        :param single_run: number associated with the run to be analyzed, eg. 15610
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param file_title: name of the figure files to be saved
        """
        stat_names = ['Mu', 'Mu error', 'Sigma', 'Sigma_error']
        folder =  self.raw_data_folder + str(int(single_run))
        run_name = os.path.basename(os.path.normpath(folder))
        print('Run: ', run_name)
        run_save = self.save_folder + '/Run ' + str(run_name) + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        mean = np.zeros((len(self.numbers), len(self.letters)))
        for i, board in enumerate(self.letters):
            run_amplitude_delta_df = self.__load_stats(single_run, board, ref_channel, variation='run')
            mean[:,i] = run_amplitude_delta_df["mu"]
        
        plot_title = f'Run {single_run}, ref {ref_channel}, mean amplitude delta'
        file_title = f'Mean amplitude delta ref {ref_channel}'
        
        plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/colormesh/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        super()._ECAL__plot_colormesh(mean, plot_title, plot_save, file_title, 'amplitude')

        
    def run_colormesh(self, ref_channel: str=None, file_title: str=None):
        """
        Plots the colormesh map with the mean amplitude over self.channel_names for every single_run in self.included_runs.
        
        :param ref_channel: name of the channel to be taken as a reference, eg. 'A1'
        :param file_title: name of the figure files to be saved
        """
        for single_run in self.included_runs:
            self.__run_colormesh_single_run(single_run, ref_channel, file_title)

