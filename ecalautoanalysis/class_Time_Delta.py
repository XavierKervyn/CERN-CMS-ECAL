""" Imports """

from .class_ECAL import *
from .class_Amplitude import * # this import is needed for the resolution

""" General function"""

def sigma_t_fit(A: float=None, *p: tuple) -> float:
    """
    Time resolution as a function of the mean amplitude A, with parameters *p to be fitted.
    
    :param A: point at which we evaluate the expression
    :param *p: pointer to the parameters N and c to be fitted
    
    :return: the time resolution for the given mean amplitude A
    """
    sigma_N = 1 # TODO: calculate actual pedestal sigma
    N, c = p
    return np.sqrt( (N*sigma_N/A)**2 + c**2 )


""" Child Class definition """

class Time_Delta(ECAL):
    """
    This class serves for the analysis of the time resolution of the detector. It has a few general private methods 
    to synchronise the different times, compute the time difference between two channels, generate the statistics
    of the data, load the data, as well as a few public getters for the mean and the standard deviation of the time delta
    and their respective errors. The other sets of metods either analyse the spills, the runs, plot colormesh of the statistics
    or study the time resolution of the system.

    :param included_runs: the list of the runs one wants to consider
    :param letters: the list of boards
    :param save_folder: folder where the csv data files are to be saved
    :param raw_data_folder: folder where the .root input reconstruction files are stored
    :param plot_save_folder: folder where the plots produced are to be staved
    """
    def __init__(self, included_runs: List[int] = None, letters: str = None,
                 save_folder: str = save_folder_global, raw_data_folder: str = raw_data_folder_global,
                 plot_save_folder: str = plot_save_folder_global, checked: bool=False):
        super().__init__(included_runs, letters, save_folder, raw_data_folder, plot_save_folder, checked)

    
    # ------------------------------------------------------------------------------------------------------------------------------
    # GENERAL
        
    def __synchroniser(self, value: float = None) -> float:
        """
        Function to remove the period shift.
        Collects the scattered peaks separated by integer multiples of the clock period to one large peak.

        :param value: list of time deltas for a single channel to synchronise
        :return: synchronised value
        """
        # How far from the center value the synchroniser should start to act. Minimum Value that makes sense physically: 0.5
        window_leniency = 0.5  
        if value > 0:
            while value > self.clock_period * window_leniency:
                value -= self.clock_period
        else:
            while value < (-self.clock_period * window_leniency):
                value += self.clock_period
        return float(Decimal(value) % Decimal(self.clock_period))

    
    def __compute_delta(self, time: pd.DataFrame = None, board: str = None, ref_channel: int = None,
                             apply_synchroniser: bool = True) -> pd.DataFrame:
        """
        Computes the time difference (delta) for a given reference channel versus the channels in the board given as argument.

        :param time: 2D dataframe containing the time for all the events considered (rows) for the channels with the given board (rows)
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param apply_synchroniser: if one wants to apply the synchronization method or not

        :return: DataFrame of the time deltas, the columns being the channels and the rows the events
        """
        time_pd = time  # Rename the column names with the channels in self._channel_names
        n_numbers = len(self.numbers)

        time_delta_pd = pd.DataFrame()

        slicing = [channel for channel in self.channel_names if channel[0] == board]
        for i, channel in enumerate(slicing):
            if channel == ref_channel:
                continue

            reference_time = time_pd[ref_channel]
            curr_time = time_pd[channel]
            time_delta = curr_time - reference_time

            # Remove period shift from the data
            if apply_synchroniser:
                time_delta = time_delta.apply(self.__synchroniser)

            time_delta = time_delta.multiply(1000)  # convert into picoseconds
            time_delta_pd[f'{channel}'] = time_delta # Save time deltas for later analysis
            
        return time_delta_pd

    
    def __generate_stats(self, single_run: int = None, board: str = None, ref_channel: str = None,
                         variation: str = None, plot: bool = False, spill_index: int = None, fit_option: str=None, nb_fits: int=None):
        """
        Creates the histograms of the time delta for a single run and single board and saves the Gaussian curve fit parameters 
        and errors mu, mu_err, sigma, sigma_err in csv files. If variation='run', only one .csv file is created, the columns 
        being the two fit parameters and their errors and the rows being the channels within the board considered. If variation='spill', 
        four .csv files are created, the columns being the channels within the board considered, and the rows are the different spills 
        within the single_run. The spill_index argument allows to consider only a given spill and thereby visualize the evolution spill
        after spill (useful feature for the DQM system). Finally, fit_option and nb_fits allow to choose a different way of computing the
        statistics.

        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param variation: either 'run' (histograms are computed over a full run) or 'spill' (separately for each spill in single_run).
        :param plot: If True, plots the histograms and fit, not if False
        :param spill_index: integer corresponding to the spill to consider, eg. 3 for the third one.
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        """
        # TODO: add path to figure to be saved
        try:
            if ref_channel not in self.channel_names:
                raise ValueError("Reference channel must be in the channel list")
                
            elif board not in self.letters:
                raise ValueError("Board must be included in the list of letters")
            
            else:
                # Computation with merged data: retrieve the amplitude
                folder = self.raw_data_folder + str(int(single_run))
                if variation == 'spill' and plot == True:
                    try:
                        h2 = uproot.concatenate({folder + f'/{spill_index}.root': 'digi'}, allow_missing=True)
                    except FileNotFoundError as e:
                        print(e)
                        return -1
                    
                else: # if variation == 'run'
                    h2 = uproot.concatenate({folder + '/*.root': 'digi'}, allow_missing=True)

                run_name = os.path.basename(os.path.normpath(folder))
                print('Run: ', run_name)
                run_save = self.save_folder + '/Run ' + run_name + '/'
                Path(run_save).mkdir(parents=True, exist_ok=True)

                slicing = [channel for channel in self.channel_names if channel[0] == board]

                ref_idx = h2[ref_channel][0]
                time = h2['time_max']
                time_pd = pd.DataFrame(time, columns=self.channel_names)

                # column header for the Dataframes
                col_list = len(self.numbers) * [board]
                col_list = [x + y for x, y in zip(col_list, self.numbers)]

                if variation == 'spill':
                    # Computation with merged data: retrieve the spill number
                    if plot:
                        h1 = uproot.concatenate({folder + f'/{spill_index}.root': 'h4'}, allow_missing=True)
                    else:
                        h1 = uproot.concatenate({folder + '/*.root': 'h4'}, allow_missing=True)
                    spill = h1['spill']
                    spill_pd = pd.DataFrame(spill, columns=["spill_nb"])

                    # merge the two Dataframes
                    tspill_pd = pd.concat([time_pd, spill_pd], axis=1, join='inner')

                    # create empty arrays to store the statistics
                    spill_set = set(tspill_pd["spill_nb"])  # set of unique spill numbers
                    time_mean_spill = np.zeros((len(spill_set), len(self.numbers)))
                    time_mean_err_spill = np.zeros((len(spill_set), len(self.numbers)))
                    time_sigma_spill = np.zeros((len(spill_set), len(self.numbers)))
                    time_sigma_err_spill = np.zeros((len(spill_set), len(self.numbers)))

                    for j, spill in enumerate(spill_set):
                        tspill_pd_temp = tspill_pd[tspill_pd.spill_nb == spill]
                        
                        if fit_option == 'synchronise':
                            apply_synchroniser = True # the time deltas will be synchronized
                        else:
                            apply_synchroniser = False 
                        time_delta_pd = self.__compute_delta(tspill_pd_temp, board, ref_channel, apply_synchroniser)

                        # 'empty' arrays to store the statistics of each channel
                        mu_arr = np.zeros(len(self.numbers))
                        mu_error_arr = np.zeros(len(self.numbers))
                        sigma_arr = np.zeros(len(self.numbers))
                        sigma_error_arr = np.zeros(len(self.numbers))

                        for i, channel in enumerate(slicing):
                            if channel == ref_channel:
                                continue

                            hist, bin_edges = np.histogram(time_delta_pd[channel], bins=self.n_bins)
                            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)
                            
                            if fit_option == 'synchronise' or fit_option == None:
                                # fitting process with one gaussian
                                mean_guess = np.average(bin_centers, weights=hist)
                                sigma_guess = np.sqrt(np.average((bin_centers - mean_guess) ** 2, weights=hist))

                                guess = [np.max(hist), mean_guess, sigma_guess]
                                try:
                                    coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=10000)
                                except RuntimeError as e:
                                    print(e)
                                    print(f"Fit unsuccessful, arbitrary coefficients set to {guess} and covariance matrix to 0.")                   
                                    coeff = guess
                                    covar = np.zeros((3,3))  
                                mu = coeff[1]
                                mu_error = np.sqrt(covar[1, 1])
                                sigma = coeff[2]
                                sigma_error = np.sqrt(covar[2, 2])
                            else:
                                if (nb_fits %2 != 1):
                                    raise ValueError("The number of fits must be odd")
                                   
                                # fitting process with multiple gaussians
                                guess = []
                                amp_list = []
                                mu_list = []
                                sigma_list = []
                                mu_error_list = []
                                sigma_error_list = []
                                
                                for index, k in enumerate(np.arange(-(nb_fits-1)/2, (nb_fits+1)/2)):
                                    mean_guess = np.average(bin_centers, weights=hist) + k*self.clock_period*1000
                                    sigma_guess = np.sqrt(np.average((bin_centers - mean_guess) ** 2, weights=hist))
                                    if k == 0:
                                        amp_guess = np.max(hist)
                                    else:
                                        amp_guess = 1
                                        
                                    guess += [amp_guess, mean_guess, sigma_guess]
                                
                                    try:
                                        coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=10000)
                                    except RuntimeError as e:
                                        print(e)
                                        print(f"Fit unsuccessful, arbitrary coefficients set to {guess} and covariance matrix to 0.")                   
                                        coeff = guess
                                        covar = np.zeros((3,3))  
                                
                                for index, k in enumerate(np.arange(-(nb_fits-1)/2, (nb_fits+1)/2)):
                                    amp_list.append(coeff[3*index]) 
                                    mu_list.append(coeff[3*index+1] - k*self.clock_period*1000)
                                    sigma_list.append(coeff[3*index+2])
                                
                                mu_error_list = [np.sqrt(covar[m,m]) for m in range(len(coeff)) if m%3 == 1]
                                sigma_error_list = [np.sqrt(covar[m,m]) for m in range(len(coeff)) if m%3 == 2]
                                
                                mu = np.average(mu_list)
                                sigma = np.average(sigma_list)
                                mu_error = np.average(mu_error_list)
                                sigma_error = np.average(sigma_error_list)

                            mu_arr[i] = mu
                            mu_error_arr[i] = mu_error
                            sigma_arr[i] = sigma
                            sigma_error_arr[i] = sigma_error

                            if plot:
                                title = f'Run: {run_name}, channel: {board + self.numbers[i]}, ref {ref_channel}, spill {spill}'
                                xlabel = 'Time delta (ps)'
                                ylabel = 'Occurrence (a.u.)'
                                
                                # save the plot
                                file_title = f'Time Delta channel {board + self.numbers[i]} ref {ref_channel} spill {spill}'
                                plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/histogram/' # make the directory
                                Path(plot_save).mkdir(parents=True, exist_ok=True)
                                super()._ECAL__plot_hist(time_delta_pd, channel, bin_centers, title, xlabel, ylabel,
                                                         plot_save, file_title, *coeff)
                                
                        time_mean_spill[j, :] = mu_arr
                        time_mean_err_spill[j, :] = mu_error_arr
                        time_sigma_spill[j, :] = sigma_arr
                        time_sigma_err_spill[j, :] = sigma_error_arr

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

                else:  # variation=='run':
                    if fit_option == 'synchronise':
                        apply_synchroniser = True # the time deltas will be synchronized
                    else:
                        apply_synchroniser = False
                    time_delta_pd = self.__compute_delta(time_pd, board, ref_channel, apply_synchroniser)

                    # 'empty' arrays to store the statistics of each channel
                    mu_arr = np.zeros(len(self.numbers))
                    mu_error_arr = np.zeros(len(self.numbers))
                    sigma_arr = np.zeros(len(self.numbers))
                    sigma_error_arr = np.zeros(len(self.numbers))

                    for i, channel in enumerate(slicing):
                        if channel == ref_channel:
                            continue

                        hist, bin_edges = np.histogram(time_delta_pd[channel], bins=1500)
                        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)

                        if fit_option == 'synchronise' or fit_option == None:
                            # fitting process with one gaussian
                            mean_guess = np.average(bin_centers, weights=hist)
                            sigma_guess = np.sqrt(np.average((bin_centers - mean_guess) ** 2, weights=hist))

                            guess = [np.max(hist), mean_guess, sigma_guess]
                            coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, maxfev=5000)
                            mu = coeff[1]
                            mu_error = np.sqrt(covar[1, 1])
                            sigma = coeff[2]
                            sigma_error = np.sqrt(covar[2, 2])
                        else:
                            if (nb_fits %2 != 1):
                                raise ValueError("The number of fits must be odd")
                            
                            # fitting process with multiple gaussians
                            guess = []
                            amp_list = []
                            mu_list = []
                            sigma_list = []
                            mu_error_list = []
                            sigma_error_list = []

                            for index, k in enumerate(np.arange(-(nb_fits-1)/2, (nb_fits+1)/2)):
                                mean_guess = np.average(bin_centers, weights=hist) + k*self.clock_period*1000
                                sigma_guess = np.sqrt(np.average((bin_centers - mean_guess) ** 2, weights=hist))
                                if k == 0:
                                    amp_guess = np.max(hist)
                                else:
                                    amp_guess = 1

                                guess += [amp_guess, mean_guess, sigma_guess]

                            coeff, covar = curve_fit(multiple_gaussians, bin_centers, hist, p0=guess, maxfev=5000)
                            
                            for index, k in enumerate(np.arange(-(nb_fits-1)/2, (nb_fits+1)/2)):
                                amp_list.append(coeff[3*index]) 
                                mu_list.append(coeff[3*index+1] - k*self.clock_period*1000)
                                sigma_list.append(coeff[3*index+2])

                            mu_error_list = [np.sqrt(covar[m,m]) for m in range(len(coeff)) if m%3 == 1]
                            sigma_error_list = [np.sqrt(covar[m,m]) for m in range(len(coeff)) if m%3 == 2]

                            mu = np.average(mu_list)
                            sigma = np.average(sigma_list)
                            mu_error = np.average(mu_error_list)
                            sigma_error = np.average(sigma_error_list)
                        
                        mu_arr[i] = mu
                        mu_error_arr[i] = mu_error
                        sigma_arr[i] = sigma
                        sigma_error_arr[i] = sigma_error
    
                        if plot:
                            title = f'Run: {run_name}, channel: {board + self.numbers[i]}, ref {ref_channel}'
                            xlabel = 'Time delta (ps)'
                            ylabel = 'Occurrence (a.u.)'
                        
                            # save the plot
                            file_title = f'Time Delta channel {board + self.numbers[i]} ref {ref_channel}'
                            plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/histogram/'
                            Path(plot_save).mkdir(parents=True, exist_ok=True) # make the directory
                            super()._ECAL__plot_hist(time_delta_pd, channel, bin_centers, title, xlabel, ylabel,
                                                     plot_save, file_title, *coeff)

                    # convert the arrays into a single Dataframe
                    run_time_delta_df = pd.DataFrame(
                        {'mu': mu_arr, 'mu error': mu_error_arr, 'sigma': sigma_arr, 'sigma error': sigma_error_arr})

                    # save it in a .csv file
                    run_time_delta_df.to_csv(self.save_folder + f'/Run {single_run}' +
                                             f'/Run time delta run {single_run} board {board} ref {ref_channel}.csv')
        except ValueError as e:
            print(e)

            
    def __load_stats(self, single_run: int = None, board: str = None, ref_channel: str = None, variation: str = None, 
                     fit_option: str="synchronise", nb_fits: int=None) -> Union[tuple, pd.DataFrame]:
        """
        Returns the Gaussian curve fit statistics of the time delta with respect to ref_channel for a single_run and board.
        If variation='run', considers the fit over the entire run, if variation='spill', considers the fit over each spill separately.

        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param variation: either 'run' (loads the statistics for the entire run) or 'spill' (for each spill).
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians

        :return: DataFrame of tuple of DataFrame of the .csv file(s) loaded (TODO: unique type?)
        """
        try:  # check if the file exists

            if variation == 'spill':  # returns a tuple with the 4 files
                return (pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                        pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill error mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                        pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill sigma time delta run {single_run} board {board} ref {ref_channel}.csv'),
                        pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill error sigma time delta run {single_run} board {board} ref {ref_channel}.csv'))
            else:  # variation=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}'
                                   + f'/Run time delta run {single_run} board {board} ref {ref_channel}.csv')
        
        except FileNotFoundError: # generating the statistics file
            print('File not found, generating .csv')
            self.__generate_stats(single_run, board, ref_channel, variation, plot=False, fit_option=fit_option, nb_fits=nb_fits)  

            # loading the file and returning it
            if variation == 'spill':  # returns a tuple with the 4 files
                return (pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                        pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill error mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                        pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill sigma time delta run {single_run} board {board} ref {ref_channel}.csv'),
                        pd.read_csv(self.save_folder + f'/Run {single_run}'
                                    + f'/Spill error sigma time delta run {single_run} board {board} ref {ref_channel}.csv'))
            else:  # variation=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}'
                                   + f'/Run time delta run {single_run} board {board} ref {ref_channel}.csv')

        except:
            raise Exception('Could not load nor generate .csv file')

            
    def get_mean(self, single_run: int=None, board: str=None, ref_channel: str=None) -> pd.core.series.Series:
        """
        Getter for the average time difference wrt. to a ref_channel for the entire channels in a board of a single run.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        
        :return: column of the dataframe containing the average time difference for each channel of the board in the rows
        """
        df = self.__load_stats(single_run, board, ref_channel, variation='run')
        return df["mu"]
    
    
    def get_sigma(self, single_run: int=None, board: str=None, ref_channel: str=None) -> pd.core.series.Series:
        """
        Getter for the standard deviation of time difference wrt. to a ref_channel for the entire channels in a board of a single run.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        
        :return: column of the dataframe containing the standard deviation of the time difference for each channel of the board in the rows
        """
        df = self.__load_stats(single_run, board, ref_channel, variation='run')
        return df["sigma"]
    
    
    def get_mean_err(self, single_run: int=None, board: str=None, ref_channel: str=None) -> pd.core.series.Series:
        """
        Getter for the error on the average time difference wrt. to a ref_channel for the entire channels in a board of a single run.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        
        :return: column of the dataframe containing the error on the average time difference for each channel of the board in the rows
        """
        df = self.__load_stats(single_run, board, ref_channel, variation='run')
        return df["mu error"]
    
    
    def get_sigma_err(self, single_run: int=None, board: str=None, ref_channel: str=None) -> pd.core.series.Series:
        """
        Getter for the error on the std dev of the time difference wrt. to a ref_channel for the entire channels in a board of a single run.
        
        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        
        :return: column of the dataframe containing the error on the std dev. of the time difference for each channel of the board in the rows
        """
        df = self.__load_stats(single_run, board, ref_channel, variation='run')
        return df["sigma error"]
            
        
    # ------------------------------------------------------------------------------------------------------------------------------
    # SPILLS

    def __spill_single_board(self, single_run: int = None, board: str = None, ref_channel: str = None, 
                             fit_option: str=None, nb_fits: int=None):
        """
        Plots the evolution over the spills in the single_run of the time delta of the channels on the board with
        respect to the ref_channel. Finally, fit_option and nb_fits allow to choose a different way of computing the
        statistics.

        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        """
        # load the Dataframes
        mean, mean_err, sigma, sigma_err = self.__load_stats(single_run, board, ref_channel, 'spill', fit_option, nb_fits)
        num_spills = mean.shape[0]  # number of spills in the single run

        slicing = [channel for channel in self.channel_names if channel[0] == board]

        # Spill column in pd.DataFrame for plot
        spill_column_tmp = [len(self.numbers) * [i] for i in range(1, num_spills + 1)]
        spill_column = []
        for lst in spill_column_tmp:
            spill_column += lst

        # Channel column in plot pd.DataFrame
        channel_column = num_spills * slicing

        # Mean and sigma columns in plot pd.DataFrame
        mean_arr = mean[slicing].to_numpy()
        mean_stacked = mean_arr.flatten()
        sigma_arr = sigma[slicing].to_numpy()
        sigma_stacked = sigma_arr.flatten()

        plot_df = pd.DataFrame(
            {"spill": spill_column, "channel": channel_column, "mean": mean_stacked, "sigma": sigma_stacked})

        xlabel = 'Spill'
        ylabel = 'Time delta (ps)'
        plot_title = f'Run {single_run}, board {board}, ref {ref_channel}, mean time delta over spills'
        
        # save the plot
        file_title = f'Time Delta board {board} ref {ref_channel}'
        plot_save = self.plot_save_folder + '/Run ' + str(single_run) + '/variation_spill/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        super()._ECAL__plot_variation(plot_df, 'spill', xlabel, ylabel, plot_title, plot_save, file_title)

        
    def __spill_single_run(self, single_run: int = None, ref_channel: str = None, all_channels: bool = None, 
                           fit_option: str=None, nb_fits: int=None):
        """
        Plots the evolution over the spills in the single_run of the time delta of the channels with respect to the ref_channel 
        on one or all the boards depending on all_channels. fit_option and nb_fits allow to choose a different way 
        of computing the statistics. 

        :param single_run: the number of a run, for example '15610'
        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: True (plots of the time delta evolution with respect to ref_channel for all boards), False (only for the board of ref_channel).
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        """
        if all_channels:
            for board in self.letters:
                self.__spill_single_board(single_run, board, ref_channel, fit_option, nb_fits)
        else:
            board = ref_channel[0]
            self.__spill_single_board(single_run, board, ref_channel, fit_option, nb_fits)

            
    def spill_variation(self, ref_channel: str = None, all_channels: bool = None, fit_option: str='synchronise', nb_fits: int=None):
        """
        Plots the evolution over the spills in each of the runs in self.included_runs of the time delta of the channels with respect 
        to the ref_channel on one or all the boards, depending on the value of all_channels. fit_option and nb_fits allow to 
        choose a different way of computing the statistics. 

        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: True (plots of the time delta evolution with respect to ref_channel for all boards), False (only for the board of ref_channel).
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        """
        for single_run in self.included_runs:
            self.__spill_single_run(single_run, ref_channel, all_channels, fit_option, nb_fits)

            
    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS

    # ---- HISTOGRAMS ----

    def __hist_single_board(self, single_run: int = None, board: str = None, ref_channel: str = None,
                                       variation: str = None, spill_i: int = None, fit_option: str=None, nb_fits: int=None):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels included in the board with 
        respect to the ref_channel for the single_run considered. fit_option and nb_fits allow to choose a different way of 
        computing the statistics. 

        :param single_run: the number of a run, for example '15610'
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param variation: either 'run' (loads the statistics for the entire run) or 'spill' (for each spill).
        :param spill_i: integer corresponding to the spill to consider, eg. 3 for the third one.
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        """
        self.__generate_stats(single_run, board, ref_channel, variation, plot=True, spill_index=spill_i, 
                              fit_option=fit_option, nb_fits=nb_fits)

        
    def __hist_single_run(self, single_run: int = None, ref_channel: str = None, all_channels: bool = None,
                                     variation: str = None, spill_i: int = None, fit_option: str=None, nb_fits: int=None):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels for each board considered 
        with respect to the ref_channel for the single_run considered.
        The boards considered can either be all of them, or only the one of ref_channel, depending on the value of all_channels.

        :param single_run: the number of a run, for example '15610'
        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: If True, plots the histograms for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        if all_channels:
            for board in self.letters:
                self.__hist_single_board(single_run, board, ref_channel, variation, spill_i, fit_option, nb_fits)
        else:
            board = ref_channel[0]
            self.__hist_single_board(single_run, board, ref_channel, variation, spill_i, fit_option, nb_fits)

            
    def hist(self, ref_channel: str = None, all_channels: bool = None, variation: str = 'run',
                        spill_i: int = None, fit_option: str='synchronise', nb_fits: int=None):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels for each board considered with 
        respect to the ref_channel for the single_run considered. The boards considered can either be all of them, or only the 
        one of ref_channel, depending on the value of all_channels. 

        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: If True, plots the histograms for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        :param variation: either 'run' (loads the statistics for the entire run) or 'spill' (for each spill).
        :param spill_i: integer corresponding to the spill to consider, eg. 3 for the third one.
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        """
        for single_run in self.included_runs:
            self.__hist_single_run(single_run, ref_channel, all_channels, variation, spill_i, fit_option, nb_fits)

            
    # ---- VARIATION OVER RUNS ----

    def __run_single_board(self, board: str = None, ref_channel: str = None, fit_option: str=None, nb_fits: int=None, file_title: str=None):
        """
        Plots the evolution over the runs of the time delta of the channels on the board with respect to the ref_channel.
        fit_option and nb_fits allow to choose how the statistics are computed (single or multiple gaussian fits). The user must also
        specify a name (file_title) for the plot to be saved, eg. 'variation 290222', or something meaningful and relevant to the 
        included_runs.

        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        :param file_title: name of the file with the saved plot
        """
        # load the Dataframes
        mean = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma = np.zeros((len(self.included_runs), len(self.numbers)))
        for i, single_run in enumerate(self.included_runs):
            run_time_delta_df = self.__load_stats(single_run, board, ref_channel, 'run', fit_option, nb_fits)  # 4 columns, n_numbers rows
            mean[i, :] = run_time_delta_df['mu']
            sigma[i, :] = run_time_delta_df['sigma']

        slicing = [channel for channel in self.channel_names if channel[0] == board]

        # Run column in pd.DataFrame for plot
        run_column_tmp = [len(self.numbers) * [i] for i in np.arange(len(self.included_runs))]
        run_column = []
        for lst in run_column_tmp:
            run_column += lst

        # Channel column in plot pd.DataFrame
        channel_column = len(self.included_runs) * slicing

        # Mean and sigma columns in plot pd.DataFrame
        mean_stacked = mean.flatten()
        sigma_stacked = sigma.flatten()

        plot_df = pd.DataFrame(
            {"run": run_column, "channel": channel_column, "mean": mean_stacked, "sigma": sigma_stacked})

        xlabel = 'Run'
        ylabel = 'Time delta (ps)'
        plot_title = f'Run {single_run}, board {board}, ref {ref_channel}, mean time delta over runs'
        
        # save the plot with the file_title specified by the user
        plot_save = self.plot_save_folder + '/run_variation/time_delta/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        super()._ECAL__plot_variation(plot_df, 'run', xlabel, ylabel, plot_title, plot_save, file_title)

        
    def run_variation(self, ref_channel: str = None, all_channels: bool = None, fit_option: str='synchronise', 
                      nb_fits: int=None, file_title: str=None):
        """
        Plots the evolution over the runs in self.included_runs of the time delta of the channels with respect to the 
        ref_channel on one or all the boards, depending on the value of all_channels. fit_option and nb_fits allow to 
        choose how the statistics are computed (single or multiple gaussian fits)

        :param ref_channel: reference channel with respect to which the differences are computed
        :param all_channels: If True, we make plots of the time delta evolution with respect to ref_channel for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        :param file_title: name of the file with the saved plot
        """
        try:
            if len(self.included_runs) <= 1:
                raise ValueError("Need at least two runs to plot a variation")
            else:
                if all_channels:
                    for board in self.letters:
                        self.__run_single_board(board, ref_channel, fit_option, nb_fits, file_title)
                else:
                    board = ref_channel[0]
                    self.__run_single_board(board, ref_channel, fit_option, nb_fits, file_title)
        except ValueError as e:
            print(e)

        
    # ---- STATISTICS OVER RUNS ----

    def __run_colormesh_single_run(self, single_run: int = None, ref_channel: str = None, fit_option: str=None, nb_fits: int=None):
        """
        Plots mean mu for the time delta with respect to ref_channel of a designated single_run in a colormesh plot.
        The mesh represents all the channels and the color represents the time delta. fit_option and nb_fits allow to 
        choose how the statistics are computed (single or multiple gaussian fits)

        :param single_run: The number of a run, for example '15484'
        :param ref_channel: reference channel with respect to which the differences are computed
        """
        
        folder = self.raw_data_folder + str(int(single_run))
        run_name = os.path.basename(os.path.normpath(folder))
        print('Run: ', run_name)
        run_save = self.save_folder + '/Run ' + str(run_name) + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        # TODO: do we also want to plot sigma, mu_err, sigma_err?
        mean = np.zeros((len(self.numbers), len(self.letters)))
        for i, board in enumerate(self.letters):
            run_time_df = self.__load_stats(single_run, board, ref_channel, 'run', fit_option, nb_fits)
            mean[:, i] = np.array(list(reversed(run_time_df["mu"])))

        plot_title = f'Run {single_run}, ref {ref_channel}, mean time delta'
        
        file_title = f"Mean Time Delta ref {ref_channel}"
        plot_save = self.plot_save_folder + '/Run ' + str(run_name) + '/colormesh/'
        Path(plot_save).mkdir(parents=True, exist_ok=True)
        super()._ECAL__plot_colormesh(mean, plot_title, plot_save, file_title)

        
    def run_colormesh(self, ref_channel: str = None, fit_option: str='synchronise', nb_fits: int=None):
        """
        Plots mean mu for the time delta with respect to ref_channel of the runs in self.included_runs in
        colormesh plots. The mesh represents all the channels and the color represents the time delta.

        :param ref_channel: reference channel with respect to which the differences are computed
        :param fit_option: if 'synchronise' or 'None', the time deltas are synchronized and one gaussian is fitted. Otherwise, the time deltas are not synchronized and multiple gaussians are fitted.
        :param nb_fits: number of gaussians if fit_option opts for multiple gaussians
        """
        for single_run in self.included_runs:
            self.__run_colormesh_single_run(single_run, ref_channel, fit_option, nb_fits)
            
    
    # ------------------------------------------------------------------------------------------------------------------------------
    # RESOLUTION

    def __resolution_single_board(self, board: str=None, ref_channel: str=None, file_title: str=None):
        """
        Computes and plots the resolution of all the channels in a given board wrt. a given ref_channel. Points are then
        fitted with the general function sigma_t_fit(). The user must provide a file_title for the plot, related to the 
        included_runs considered for the resolution.
        
        :param board: board considered
        :param ref_channel: reference channel with respect to which the differences are computed
        :param file_title: name of the file with the saved plot
        """
        a = Amplitude(self.included_runs, self.letters, checked=True) 
        
        # mean amplitude, sigma and associated errors for the chosen board
        A_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        A_err_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma_err_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        
        # mean amplitude and associated error for the board containing the reference channel
        A_ref_lst = np.zeros((len(self.included_runs), len(self.numbers)))
        A_ref_err_lst = np.zeros((len(self.included_runs), len(self.numbers)))

        # fill the lists
        for i, single_run in enumerate(self.included_runs):
            A_lst[i] = a.get_mean(single_run, board)
            sigma_lst[i] = np.abs(self.get_sigma(single_run, board, ref_channel))
            A_err_lst[i] = a.get_mean_err(single_run, board)
            sigma_err_lst[i] = np.abs(self.get_sigma_err(single_run, board, ref_channel))

            A_ref_lst[i] = a.get_mean(single_run, ref_channel[0])
            A_ref_err_lst[i] = a.get_mean_err(single_run, ref_channel[0])

        # keep only the data for the ref_channel
        A2 = A_ref_lst[:,int(ref_channel[1])-1]
        dA2 = A_ref_err_lst[:,int(ref_channel[1])-1]

        for j, channel in enumerate([board+number for number in self.numbers]):
            if channel == ref_channel: # ignore the reference channel if it is in the board given
                continue
            A1 = A_lst[:,j]
            dA1 = A_err_lst[:,j]
            x = A1*A2 / np.sqrt(A1**2 + A2**2)

            # define the errorbars
            xerror = x * (dA1/A1 + dA2/A2 + (A1*dA1 + A2*dA2)/(A1**2 + A2**2)) # 'mean' amplitude across channels 
            yerror = sigma_err_lst[:,j]

            guess = [100000, 50] # provide a good first guess to compute the fit
            coeff, covar = curve_fit(sigma_t_fit, x, sigma_lst[:,j], sigma=yerror, p0=guess, maxfev=5000) # fit the coefficients to the data

            # create the plotly figure
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            df_data = pd.DataFrame({'x': x, 'y': sigma_lst[:,j], "err_x": xerror, "err_y": yerror})
            trace1 = px.scatter(data_frame=df_data, x='x', y='y', 
                                error_x="err_x", error_y="err_y", 
                                color_discrete_sequence=["Crimson"], 
                                labels={"y": r"$\sigma_{\Delta t}$ (ps)"})
            fig.add_trace(trace1.data[0]) # add the trace to the figure
            
            xx = np.linspace(np.min(x), np.max(x)) # linspace to have an almost continuous fit
            df_fit = pd.DataFrame({'x': xx, 'y': sigma_t_fit(xx, *coeff)})
            trace2 = px.line(df_fit, x='x', y='y')
            fig.add_trace(trace2.data[0], secondary_y=False) # plot the fit
            
            fig.add_annotation(text=f'Parameters: N={round(coeff[0],2)}, c={round(coeff[1],2)}', xref='x domain', yref='y domain', x=1, y=1)

            # add title and label
            plot_title = f"Time delta absolute resolution, ref {ref_channel}, channel {channel}"
            xlabel = "Average amplitude A (ADC counts)"
            ylabel = "Absolute time resolution (ps)"
            
            fig.update_layout(title=plot_title,
                              xaxis=dict(title=xlabel),
                              yaxis=dict(title=ylabel),
                              updatemenus=[ # add the option to change the scale of the axis to linear, semilogy or loglog
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
                             );

            plot_save = self.plot_save_folder + '/resolution/time_delta/'
            Path(plot_save).mkdir(parents=True, exist_ok=True)
            
            path = plot_save
            fig.write_image(path + file_title + f' ref {ref_channel} channel {channel}' + '.png')
            fig.write_image(path + file_title + f' ref {ref_channel} channel {channel}' + '.pdf')
            fig.write_image(path + file_title + f' ref {ref_channel} channel {channel}' + '.svg')
            fig.write_html(path + file_title + f' ref {ref_channel} channel {channel}' + '.html')


    def resolution(self, ref_channel: str=None, file_title: str=None):
        """
        Computes and plots the resolution of all board wrt. a given ref_channel. Points are then
        fitted with the general function sigma_t_fit(). The user must provide a file_title for the plot, related to the 
        included_runs considered for the resolution.
        
        :param ref_channel: reference channel with respect to which the differences are computed
        :param file_title: name of the file with the saved plot
        """
        try:
            if len(self.included_runs) <= 1:
                raise ValueError("Need at least two runs to plot a resolution")    
            for board in self.letters:
                print(f'Board {board}')
                self.__resolution_single_board(board, ref_channel, file_title)
        except ValueError as e:
            print(e)
            
