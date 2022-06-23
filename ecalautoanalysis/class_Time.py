""" Imports """

from .class_ECAL import *


""" 2nd Child Class definition """     
        
class Time(ECAL):
    """
    Class for the analysis of the time resolution of the detector
    
    :param included_runs: the list of the runs one wants to consider
    :param letters: the list of boards
    :param save_folder: folder where the csv data files are to be saved
    :param raw_data_folder: folder where the .root input reconstruction files are stored
    :param plot_save_folder: folder where the plots produced are to be staved
    """
    def __init__(self, included_runs: List[int]=None, letters: str=None,
                 save_folder: str=save_folder_global, raw_data_folder: str=raw_data_folder_global,
                 plot_save_folder: str=plot_save_folder_global):   
        super().__init__(included_runs, letters, save_folder, raw_data_folder, plot_save_folder)
    
    
    def __synchroniser(self, value: float=None) -> float:
        """ 
        Function to remove the period shift.
        Collects the scattered peaks separated by integer multiples of the clock period to one large peak.
        
        :param value: list of time deltas for a single channel to synchronize 
        :return: synchronized value
        """
        clock_period = 6.238 # nanoseconds    
        window_leniency = 0.5 # How far from the center value the synchroniser should start to act. Minimum Value that makes sense physically: 0.5
        if value > 0:
            while value > clock_period * window_leniency:
                value -= clock_period
        else:
            while value < (-clock_period * window_leniency):
                value += clock_period
        return float(Decimal(value) % Decimal(clock_period))
    
    
    def __to_channel_converter(self, channel_number: int=None) -> str:
        """
        Converts the channel number to the appropriate Channel. For example 7 -> 'B3'.
        
        :param channel_number: index of the channel considered in self.channel_names
        :return: channel number
        """
        board_counter = 0
        while channel_number > 4:
            board_counter += 1
            channel_number -= 5
        return f'{self.letters[board_counter]}{self.numbers[channel_number]}'

    
    def __compute_time_delta(self, time: pd.DataFrame=None, board: str=None, ref_channel: int=None, apply_synchroniser: bool=True) -> pd.DataFrame:
        """
        Computes the time difference (delta) for a given reference channel versus the channels in the board given.
        
        :param time: 2D dataframe containing the time for all the events considered (rows) for the channels with the given board (rows)
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param apply_synchronization: if one wants to apply the synchronization method or not
        
        :return: DataFrame of the time deltas, the columns being the channels and the rows the events
        """
        
        # Rename the column names with the channels in self._channel_names
        time_pd = time
        n_numbers = len(self.numbers)

        time_delta_pd = pd.DataFrame()

        slicing = [channel for channel in self.channel_names if channel[0]==board]
        for i, channel in enumerate(slicing):
            if channel == ref_channel:
                continue

            reference_time = time_pd[ref_channel]
            curr_time = time_pd[channel]
            time_delta = curr_time - reference_time

            # Remove period shift from the data
            if apply_synchroniser: 
                time_delta = time_delta.apply(self.__synchroniser)

            time_delta = time_delta.multiply(1000) # TODO: why?

            # Save time deltas for later analysis
           
            time_delta_pd[f'{channel}'] = time_delta
        return time_delta_pd
        
        
    def __generate_stats(self, single_run: int=None, board: str=None, ref_channel: str=None, variation: str=None, plot: bool=False):
        """ 
        Creates the histograms of the time delta for a single run and single board and saves the Gaussian curve fit parameters and errors mu, mu_err, sigma, sigma_err in csv files.
        If variation='run', only one .csv file is created, the columns being the two fit parameters and their errors and the rows being the channels within the board considered.
        If variation='spill', four .csv files are created, the columns being the channels within the board considered, and the rows are the different spills within the single_run.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param variation: either 'run' or 'spill'. If 'run', the histograms are computed over a full run, if 'spill' the histograms are computed separately for each spill in single_run.       
        :param plot: If True, plots the histograms and fit, not if False
        """
        try:
            if ref_channel not in self.channel_names:
                raise ValueError("Reference channel must be in the channel list")

            else:
                # Computation with merged data
                folder =  self.raw_data_folder + str(int(single_run))
                h2 = uproot.concatenate({folder+'/*.root' : 'digi'}, allow_missing = True)

                run_name = os.path.basename(os.path.normpath(folder))
                print('Run: ', run_name)
                run_save = self.save_folder + '/Run ' + run_name + '/'
                Path(run_save).mkdir(parents=True, exist_ok=True)

                slicing = [channel for channel in self.channel_names if channel[0]==board]

                ref_idx = h2[ref_channel][0]
                time = h2['time_max']
                time_pd = pd.DataFrame(time, columns=self.channel_names)

                # column header for the Dataframes
                col_list = len(self.numbers)*[board]; col_list = [x + y for x,y in zip(col_list, self.numbers)]

                if variation=='spill':
                    # Computation with merged data: retrieve the spill number
                    h1 = uproot.concatenate({folder + '/*.root' : 'h4'}, allow_missing = True)
                    spill = h1['spill'] 
                    spill_pd = pd.DataFrame(spill, columns=["spill_nb"]) 

                    # merge the two Dataframes
                    tspill_pd = pd.concat([time_pd, spill_pd], axis=1, join='inner')

                    # create empty arrays to store the statistics
                    spill_set = set(tspill_pd["spill_nb"]) # set of unique spill numbers
                    time_mean_spill = np.zeros((len(spill_set), len(self.numbers)))
                    time_mean_err_spill = np.zeros((len(spill_set), len(self.numbers)))
                    time_sigma_spill = np.zeros((len(spill_set), len(self.numbers)))
                    time_sigma_err_spill = np.zeros((len(spill_set), len(self.numbers)))

                    for j, spill in enumerate(spill_set):
                        tspill_pd_temp = tspill_pd[tspill_pd.spill_nb == spill]

                        time_delta_pd = self.__compute_time_delta(tspill_pd_temp, board, ref_channel, apply_synchroniser=True)

                        # 'empty' arrays to store the statistics of each channel
                        mu_arr = np.zeros(len(self.numbers))
                        mu_error_arr = np.zeros(len(self.numbers))
                        sigma_arr = np.zeros(len(self.numbers))
                        sigma_error_arr = np.zeros(len(self.numbers))

                        for i, channel in enumerate(slicing):
                            if channel == ref_channel:
                                continue

                            hist, bin_edges = np.histogram(time_delta_pd[channel], bins = 1500)
                            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)

                            # fitting process
                            mean_guess = np.average(bin_centers, weights=hist)
                            sigma_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist))

                            guess = [np.max(hist), mean_guess, sigma_guess]
                            coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=5000)
                            mu = coeff[1]
                            mu_error = np.sqrt(covar[1,1])
                            sigma = coeff[2]
                            sigma_error = np.sqrt(covar[2,2])
                            mu_arr[i] = mu
                            mu_error_arr[i] = mu_error
                            sigma_arr[i] = sigma
                            sigma_error_arr[i] = sigma_error

                            if plot: # TODO: add path name to save the plots
                                title = f'Run: {run_name}, Channel: {board+self.numbers[i]}, Ref {ref_channel}, Spill {spill}'
                                xlabel = 'Time delta (ps)'
                                ylabel = 'Occurence (a.u.)'
                                path = ''
                                super()._ECAL__plot_hist(time_delta_pd, channel, bin_centers, title, xlabel, ylabel, path, *coeff)

                        time_mean_spill[j,:] = mu_arr
                        time_mean_err_spill[j,:] = mu_error_arr
                        time_sigma_spill[j,:] = sigma_arr
                        time_sigma_err_spill[j,:] = sigma_error_arr

                    # convert the matrices to Dataframes
                    spill_time_mean_df = pd.DataFrame(time_mean_spill, columns=col_list)
                    spill_time_mean_err_df = pd.DataFrame(time_mean_err_spill, columns=col_list)
                    spill_time_sigma_df = pd.DataFrame(time_sigma_spill, columns=col_list)
                    spill_time_sigma_err_df = pd.DataFrame(time_sigma_err_spill, columns=col_list)

                    # save these in .csv files
                    spill_time_mean_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                              + f'/Spill mean time delta run {single_run} board {board} ref {ref_channel}.csv')
                    spill_time_mean_err_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                                  + f'/Spill error mean time delta run {single_run} board {board} ref {ref_channel}.csv')
                    spill_time_sigma_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                               + f'/Spill sigma time delta run {single_run} board {board} ref {ref_channel}.csv')
                    spill_time_sigma_err_df.to_csv(self.save_folder + f'/Run {single_run}' 
                                                   + f'/Spill error sigma time delta run {single_run} board {board} ref {ref_channel}.csv')

                else: # variation=='run':
                    time_delta_pd = self.__compute_time_delta(time_pd, board, ref_channel, apply_synchroniser=True)

                    # 'empty' arrays to store the statistics of each channel
                    mu_arr = np.zeros(len(self.numbers))
                    mu_error_arr = np.zeros(len(self.numbers))
                    sigma_arr = np.zeros(len(self.numbers))
                    sigma_error_arr = np.zeros(len(self.numbers))

                    for i, channel in enumerate(slicing):
                        if channel == ref_channel:
                            continue

                        hist, bin_edges = np.histogram(time_delta_pd[channel], bins = 1500)
                        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  

                        # fitting process
                        mean_guess = np.average(bin_centers, weights=hist)
                        sigma_guess = np.sqrt(np.average((bin_centers - mean_guess)**2, weights=hist))

                        guess = [np.max(hist), mean_guess, sigma_guess]
                        coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=5000)
                        mu = coeff[1]
                        mu_error = np.sqrt(covar[1,1])
                        sigma = coeff[2]
                        sigma_error = np.sqrt(covar[2,2])
                        mu_arr[i] = mu
                        mu_error_arr[i] = mu_error
                        sigma_arr[i] = sigma
                        sigma_error_arr[i] = sigma_error

                        if plot: # TODO: add path name to save the plots
                            title = f'Run: {run_name}, Channel: {board+self.numbers[i]}, Ref {ref_channel}'
                            xlabel = 'Time delta (ps)'
                            ylabel = 'Occurence (a.u.)'
                            path = ''
                            super()._ECAL__plot_hist(time_delta_pd, channel, bin_centers, title, xlabel, ylabel, path, *coeff)

                    # convert the arrays into a single Dataframe
                    run_time_delta_df = pd.DataFrame({'mu':mu_arr, 'mu error':mu_error_arr, 'sigma': sigma_arr, 'sigma error': sigma_error_arr})

                    # save it in a .csv file
                    run_time_delta_df.to_csv(self.save_folder + f'/Run {single_run}' + 
                                             f'/Run time delta run {single_run} board {board} ref {ref_channel}.csv')
        except ValueError as e:
            print(e)

            
    def __load_stats(self, single_run: int=None, board: str=None, ref_channel: str=None, variation: str=None) -> Union[tuple, pd.DataFrame]:
        """
        Returns the Gaussiant curve fit statistics of the time delta with respect to ref_channel for a single_run and board.
        If variation='run', considers the fit over the entire run, if variation='spill', considers the fit over each spill separately.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param variation: either 'run' or 'spill'. If 'run', loads the statistics for the entire run, if 'spill' loads the statistics for each spill.
        
        :return: DataFrame of tuple of DataFrame of the .csv file(s) loaded (TODO: unique type?)
        """
        try: # check if the file exists
            
            if variation=='spill': # returns a tuple with the 4 files
                return (pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                    + f'/Spill mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                + f'/Spill error mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                + f'/Spill sigma time delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                + f'/Spill error sigma time delta run {single_run} board {board} ref {ref_channel}.csv'))
            else: # variation=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                   + f'/Run time delta run {single_run} board {board} ref {ref_channel}.csv')
        except FileNotFoundError:
            print('File not found, generating .csv')
            self.__generate_stats(single_run, board, ref_channel, variation, plot=False) # generating the statistics file
            
            # loading the file and returning it
            if variation=='spill': # returns a tuple with the 4 files
                return (pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                    + f'/Spill mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                + f'/Spill error mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                + f'/Spill sigma time delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                + f'/Spill error sigma time delta run {single_run} board {board} ref {ref_channel}.csv'))
            else: # variation=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}' 
                                   + f'/Run time delta run {single_run} board {board} ref {ref_channel}.csv')
        
        except: 
            raise Exception('Could not load nor generate .csv file')

            
    # ------------------------------------------------------------------------------------------------------------------------------
    # SPILLS
    
    def __time_delta_spill_single_board(self, single_run: int=None, board: str=None, ref_channel: str=None): 
        """
        Plots the evolution over the spills in the single_run of the time delta of the channels on the board with respect to the ref_channel.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
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
        ylabel = 'Time delta (ps)'
        plot_title = f'Run {single_run}, Board {board}, Ref {ref_channel}, mean time delta over spills'
        
        super()._ECAL__plot_variation(plot_df, 'spill', xlabel, ylabel, plot_title)
        

    def __time_delta_spill_single_run(self, single_run: int=None, ref_channel: str=None, all_channels: bool=None):
        """
        Plots the evolution over the spills in the single_run of the time delta of the channels with respect to the ref_channel on one or all the boards depending on all_channels.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: If True, we make plots of the time delta evolution with respect to ref_channel for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        if all_channels:
            for board in self.letters:
                self.__time_delta_spill_single_board(single_run, board, ref_channel)
        else:
            board = ref_channel[0]
            self.__time_delta_spill_single_board(single_run, board, ref_channel)
            
    
    def variation_time_delta_spill(self, ref_channel: str=None, all_channels: bool=None):
        """
        Plots the evolution over the spills in each of the runs in self.included_runs of the time delta of the channels with respect to the ref_channel on one or all the boards, depending on the value of all_channels.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: If True, we make plots of the time delta evolution with respect to ref_channel for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        
        for single_run in self.included_runs:
            self.__time_delta_spill_single_run(single_run, ref_channel, all_channels)
    
    
    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS
    
    # ---- HISTOGRAMS ----
    
    def __hist_time_delta_single_board(self, single_run: int=None, board: str=None, ref_channel: str=None):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels included in the board with respect to the ref_channel for the single_run considered.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        """
        self.__generate_stats(single_run, board, ref_channel, variation='run', plot=True)
        
        
    def __hist_time_delta_single_run(self, single_run: int=None, ref_channel: str=None, all_channels: bool=None):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels for each board considered with respect to the ref_channel for the single_run considered.
        The boards considered can either be all of them, or only the one of ref_channel, depending on the value of all_channels. 
        
        :param single_run: the number of a run, for example '15610'
        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: If True, plots the histograms for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        if all_channels:
            for board in self.letters:
                self.__hist_time_delta_single_board(single_run, board, ref_channel)
        else:
            board = ref_channel[0]
            self.__hist_time_delta_single_board(single_run, board, ref_channel)
          
        
    def hist_time_delta(self, ref_channel: str=None, all_channels: bool=None):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels for each board considered with respect to the ref_channel for the single_run considered.
        The boards considered can either be all of them, or only the one of ref_channel, depending on the value of all_channels. 
        
        :param single_run: the number of a run, for example '15610'
        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: If True, plots the histograms for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        for single_run in self.included_runs:
            self.__hist_time_delta_single_run(single_run, ref_channel, all_channels)

            
    # ---- VARIATION OVER RUNS ----
    
    def __time_delta_run_single_board(self, board: str=None, ref_channel: str=None):
        """
        Plots the evolution over the runs of the time delta of the channels on the board with respect to the ref_channel.
        
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        """
        
        # load the Dataframes     
        mean = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma = np.zeros((len(self.included_runs), len(self.numbers)))
        for i, single_run in enumerate(self.included_runs):
            run_time_delta_df = self.__load_stats(single_run, board, ref_channel, 'run') # 4 columns, n_numbers rows
            mean[i,:] = run_time_delta_df["mu"]
            sigma[i,:] = run_time_delta_df["sigma"] 
        
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
        ylabel = 'Time delta (ps)'
        plot_title = f'Run {single_run}, Board {board}, Ref {ref_channel}, mean time delta over runs'
        
        super()._ECAL__plot_variation(plot_df, 'run', xlabel, ylabel, plot_title)
        

    def variation_time_delta_run(self, ref_channel: str=None, all_channels: bool=None):
        """
        Plots the evolution over the runs in self.included_runs of the time delta of the channels with respect to the ref_channel on one or all the boards, depending on the value of all_channels.
        
        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: If True, we make plots of the time delta evolution with respect to ref_channel for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        try:
            if len(self.included_runs) <= 1:
                raise ValueError("Need at least two runs to plot a variation")
            else:
                if all_channels:
                    for board in self.letters:
                        self.__time_delta_run_single_board(board, ref_channel)
                else:
                    board = ref_channel[0]
                    self.__time_delta_run_single_board(board, ref_channel)
        except ValueError as e:
            print(e)
    
    # ---- STATISTICS OVER RUNS ----
    
    def __run_statistics_single_run(self, single_run: int=None, ref_channel: str=None):
        """ 
        Plots mean mu for the time delta with respect to ref_channel of a designated single_run in a colormesh plot.
        The mesh represents all the channels and the color reperesents the time delta.
    
        :param single_run: The number of a run, for example '15484'
        :param ref_channel: reference channel with respect to which the differences are computed
        """
        stat_names = ['Mu', 'Mu error', 'Sigma', 'Sigma_error']
        folder =  self.raw_data_folder + str(int(single_run))
        run_name = os.path.basename(os.path.normpath(folder))
        print('Run: ', run_name)
        run_save = self.save_folder + '/Run ' + str(run_name) + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        # TODO: do we also want to plot sigma, mu_err, sigma_err?
        mean = np.zeros((len(self.numbers), len(self.letters)))
        for i, board in enumerate(self.letters):
            run_time_df = self.__load_stats(single_run, board, ref_channel, 'run')
            mean[:,i] = np.array(list(reversed(run_time_df["mu"])))
    
        plot_title = f'Run {single_run}, Board {board}, Ref {ref_channel}, mean time delta over runs'
        super()._ECAL__plot_colormesh(mean, plot_title)

        
    def run_statistics(self, ref_channel: str=None):
        """ 
        Plots mean mu for the time delta with respect to ref_channel of the runs in self.included_runs in colormesh plots.
        The mesh represents all the channels and the color reperesents the time delta.
    
        :param ref_channel: reference channel with respect to which the differences are computed
        """
        for single_run in self.included_runs:
            self.__run_statistics_single_run(single_run, ref_channel)
