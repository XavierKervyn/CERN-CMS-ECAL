""" Imports """

from class_ECAL import *


""" 2nd Child Class definition """     
        
class Time(ECAL):
    """
    Class for the analysis of the time resolution of the detector
    
    included_runs -- (list of integers): the list of the runs one wants to consider
    letters -- (list of strings): the list of boards
    save_folder -- (string): folder where the csv data files are to be saved
    raw_data_folder -- (string): folder where the .root input reconstruction files are stored
    plot_save_folder -- (string): folder where the plots produced are to be staved
    """
    def __init__(self, included_runs, letters, # TODO: check if h5 file generated for the runs
                 save_folder = save_folder_global, raw_data_folder = raw_data_folder_global,
                 plot_save_folder = plot_save_folder_global):   
        super().__init__(included_runs, letters, save_folder, raw_data_folder, plot_save_folder)
    
    
    def __synchroniser(self, value):
        """ 
        Function to remove the period shift. 
        Collects the scattered peaks separated by integer multiples of the clock period to one large peak.
        
        value -- (list of float): list of time deltas for a single channel to synchronize 
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
    
    
    def __to_channel_converter(self, channel_number):
        """
        Converts the channel number to the appropriate Channel. For example 7 -> 'B3'.
        
        channel_number (int): index of the channel considered in self.channel_names
        """
        board_counter = 0
        while channel_number > 4:
            board_counter += 1
            channel_number -= 5
        return f'{self.letters[board_counter]}{self.numbers[channel_number]}'

    
    def __compute_time_delta(self, time, board, ref_channel, apply_synchroniser=True):
        """
        Computes the time difference (delta) for a given reference channel versus the channels in the board given.
        
        time -- (pd.DataFrame): 2D dataframe containing the time for all the events considered (rows) for the channels with
                                the given board (rows)
        board -- (string): board considered
        ref_channel -- (string): reference channel with respect to which the differences are computed
        apply_synchronization -- (bool): if one wants to apply the synchronization method or not
        """
        
        time_pd = pd.DataFrame(time, columns=self.channel_names)
        n_numbers = len(self.numbers)

        time_delta_pd = pd.DataFrame()

        slicing = [channel for channel in self.channel_names if channel[0]==board]
        for i, channel in enumerate(slicing):
            if channel == ref_channel:
                continue

            reference_time = time_pd[ref_channel]
            curr_time = time_pd[channel]
            time_delta = curr_time - reference_time # TODO: check sign

            # Remove period shift from the data
            if apply_synchroniser: 
                time_delta = time_delta.apply(self.__synchroniser)

            time_delta = time_delta.multiply(1000) # TODO: why?

            # Save time deltas for later analysis
           
            time_delta_pd[f'{channel}'] = time_delta
        return time_delta_pd
        
        
    def __generate_stats(self, single_run, board, ref_channel, param='run', plot=False):
        """ 
        Creates the histograms of the time delta for a single run and single board and saves the Gaussian curve fit parameters and errors mu, mu_err, sigma, sigma_err in csv files.
        If param='run', only one .csv file is created, the columns being the two fit parameters and their errors and the rows being the channels within the board considered.
        If param='spill', four .csv files are created, the columns being the channels within the board considered, and the rows are the different spills within the single_run.
        
        single_run -- (string or int): the number of a run, for example '15610'
        board -- (string): board considered
        ref_channel -- (string): reference channel with respect to which the differences are computed
        param -- (string): either 'run' or 'spill'. If 'run', the histograms are computed over a full run, 
                           if 'spill' the histograms are computed separately for each spill in single_run.
                           
        plot -- (bool): If True, plots the histograms and fit, not if False
        """

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
        print(time)
        time_pd = pd.DataFrame(time, columns=self.channel_names)
        
        # column header for the Dataframes
        col_list = len(self.numbers)*[board]; col_list = [x + y for x,y in zip(col_list, self.numbers)]
        
        if param=='spill':
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
                    border_size = 2000 # TODO: is not useful?
                    
                    if plot:
                        plt.figure()
                        hist, bin_edges, _ = plt.hist(time_delta_pd[channel], bins = 1500, label="Amplitude Histogram")
                    else:
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
                    
                    if plot:
                        # plotting the histogram with a gaussian fit, the mean and the standard deviation
                        plt.plot(bin_centers, gaussian(bin_centers, *coeff), label='Gaussian Fit')
                        plt.axvline(mu, label = f'Mean: {np.around(mu, decimals = 1)} ps', color = 'red')
                        sigma_color = 'pink'
                        plt.axvline(mu + sigma, label = f'Std Dev: {np.around(sigma, decimals = 1)} ps', color = sigma_color)
                        plt.axvline(mu - sigma, color = sigma_color)

                        plt.title(f'Run: {run_name}, Ref: {ref_channel}, Channel: {board+self.numbers[i]}, Spill {spill}')
                        plt.xlabel('Time delta (ps)')
                        plt.ylabel('Occurence (a.u.)')
                        plt.legend(loc='best')

                        plt.show()
                    
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
            spill_time_mean_df.to_csv(self.save_folder + f'/Spill mean time delta run {single_run} board {board} ref {ref_channel}.csv')
            spill_time_mean_err_df.to_csv(self.save_folder + f'/Spill error mean time delta run {single_run} board {board} ref {ref_channel}.csv')
            spill_time_sigma_df.to_csv(self.save_folder + f'/Spill sigma time delta run {single_run} board {board} ref {ref_channel}.csv')
            spill_time_sigma_err_df.to_csv(self.save_folder + f'/Spill error sigma time delta run {single_run} board {board} ref {ref_channel}.csv')
                
        elif param=='run':
            time_delta_pd = self.__compute_time_delta(time_pd, board, ref_channel, apply_synchroniser=True)
            
            # 'empty' arrays to store the statistics of each channel
            mu_arr = np.zeros(len(self.numbers))
            mu_error_arr = np.zeros(len(self.numbers))
            sigma_arr = np.zeros(len(self.numbers))
            sigma_error_arr = np.zeros(len(self.numbers))

            for i, channel in enumerate(slicing):
                if channel == ref_channel:
                    continue
                border_size = 2000 # TODO: is not useful?

                if plot:
                    plt.figure()
                    hist, bin_edges, _ = plt.hist(time_delta_pd[channel], bins = 1500, label="Amplitude Histogram")
                else:
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
                
                if plot:
                        # plotting the histogram with a gaussian fit, the mean and the standard deviation
                        plt.plot(bin_centers, gaussian(bin_centers, *coeff), label='Gaussian Fit')
                        plt.axvline(mu, label = f'Mean: {np.around(mu, decimals = 1)} ps', color = 'red')
                        sigma_color = 'pink'
                        plt.axvline(mu + sigma, label = f'Std Dev: {np.around(sigma, decimals = 1)} ps', color = sigma_color)
                        plt.axvline(mu - sigma, color = sigma_color)

                        plt.title(f'Run: {run_name}, Ref: {ref_channel}, Channel: {board+self.numbers[i]}')
                        plt.xlabel('Time delta (ps)')
                        plt.ylabel('Occurence (a.u.)')
                        plt.legend(loc='best')

                        plt.show()

            # convert the arrays into a single Dataframe
            run_time_delta_df = pd.DataFrame({'mu':mu_arr, 'mu error':mu_error_arr, 'sigma': sigma_arr, 'sigma error': sigma_error_arr})

            # save it in a .csv file
            run_time_delta_df.to_csv(self.save_folder + f'/Run time delta run {single_run} board {board} ref {ref_channel}.csv')
        
        else: # TODO: throw exception
            print('wrong parameter, either spill or run')
            
            
    def __load_stats(self, single_run, board, ref_channel, param):
        """
        Returns the Gaussiant curve fit statistics of the time delta with respect to ref_channel for a single_run and board.
        If param='run', considers the fit over the entire run, if param='spill', considers the fit over each spill separately.
        
        single_run -- (string or int): the number of a run, for example '15610'
        board -- (string): board considered
        ref_channel -- (string): reference channel with respect to which the differences are computed
        param -- (string): either 'run' or 'spill'. If 'run', loads the statistics for the entire run, 
                           if 'spill' loads the statistics for each spill.
        """
        # TODO: throw exception or generate file if file does not exist
        
        # TODO: remove when exception implemented
        self.__generate_stats(single_run, board, ref_channel, param) # generating the statistics file
        
        # loading the file and returning it
        if param=='spill': # returns a tuple with the 4 files
            return (pd.read_csv(self.save_folder + f'/Spill mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                pd.read_csv(self.save_folder + f'/Spill error mean time delta run {single_run} board {board} ref {ref_channel}.csv'),
                pd.read_csv(self.save_folder + f'/Spill sigma time delta run {single_run} board {board} ref {ref_channel}.csv'),
                pd.read_csv(self.save_folder + f'/Spill error sigma time delta run {single_run} board {board} ref {ref_channel}.csv'))
        
        elif param=='run':
            return pd.read_csv(self.save_folder + f'/Run time delta run {single_run} board {board} ref {ref_channel}.csv')
        
        else:
            # TODO: throw exception
            print('wrong parameter, either spill/run')

            
    # ------------------------------------------------------------------------------------------------------------------------------
    # SPILLS
    def __time_delta_spill_single_board(self, single_run, board, ref_channel): 
        """
        Plots the evolution over the spills in the single_run of the time delta of the channels on the board with respect to the ref_channel.
        
        single_run -- (string or int): the number of a run, for example '15610'
        board -- (string): board considered
        ref_channel -- (string): reference channel with respect to which the differences are computed
        """
        
        # load the Dataframes
        mean, mean_err, sigma, sigma_err = self.__load_stats(single_run, board, ref_channel, 'spill')
        num_spills = mean.shape[0] # number of spills in the single run
        
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        
        # Plotting evolution for a single board
        plt.figure()
        for i, number in enumerate(self.numbers):
            plt.errorbar(np.arange(num_spills), mean[slicing[i]], yerr=sigma[slicing[i]], label=board+number)
            
        plt.xticks(np.arange(num_spills), 1+np.arange(num_spills)) # TODO: check
        plt.legend(loc='best')
        plt.title(f'Run {single_run}, board {board}, ref {ref_channel}, mean time delta over spills')
        plt.xlabel('Spill')
        plt.ylabel('Time delta (ps)')
        plt.show()
        

    def __time_delta_spill_single_run(self, single_run, ref_channel, all_channels):
        """
        Plots the evolution over the spills in the single_run of the time delta of the channels with respect to the ref_channel on one or all the boards depending on all_channels.
        
        single_run -- (string or int): the number of a run, for example '15610'
        board -- (string): board considered
        ref_channel -- (string): reference channel with respect to which the differences are computed
        all_channels -- (bool): If True, we make plots of the time delta evolution with respect to ref_channel for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        if all_channels:
            for board in self.letters:
                self.__time_delta_spill_single_board(single_run, board, ref_channel)
        else:
            board = ref_channel[0]
            self.__time_delta_spill_single_board(single_run, board, ref_channel)
            
    
    def variation_time_delta_spill(self, ref_channel, all_channels):
        """
        Plots the evolution over the spills in each of the runs in self.included_runs of the time delta of the channels with respect to the ref_channel on one or all the boards, depending on the value of all_channels.
        
        single_run -- (string or int): the number of a run, for example '15610'
        board -- (string): board considered
        ref_channel -- (string): reference channel with respect to which the differences are computed
        all_channels -- (bool): If True, we make plots of the time delta evolution with respect to ref_channel for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        
        for single_run in self.included_runs:
            self.__time_delta_spill_single_run(single_run, ref_channel, all_channels)
    
    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS
    
    
    # ---- HISTOGRAMS ----
    def __hist_time_delta_single_board(self, single_run, board, ref_channel):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels included in the board with respect to the ref_channel for the single_run considered.
        
        single_run -- (string or int): the number of a run, for example '15610'
        board -- (string): board considered
        ref_channel -- (string): reference channel with respect to which the differences are computed
        """
        self.__generate_stats(single_run, board, ref_channel, param='run', plot=True)
        
        
    def __hist_time_delta_single_run(self, single_run, ref_channel, all_channels):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels for each board considered with respect to the ref_channel for the single_run considered.
        The boards considered can either be all of them, or only the one of ref_channel, depending on the value of all_channels. 
        
        single_run -- (string or int): the number of a run, for example '15610'
        ref_channel -- (string): reference channel with respect to which the differences are computed
        all_channels -- (bool): If True, plots the histograms for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        if all_channels:
            for board in self.letters:
                self.__hist_time_delta_single_board(single_run, board, ref_channel)
        else:
            board = ref_channel[0]
            self.__hist_time_delta_single_board(single_run, board, ref_channel)
          
        
    def hist_time_delta(self, ref_channel, all_channels):
        """
        Plots the histograms and corresponding Gaussian fits of the time delta of the channels for each board considered with respect to the ref_channel for the single_run considered.
        The boards considered can either be all of them, or only the one of ref_channel, depending on the value of all_channels. 
        
        single_run -- (string or int): the number of a run, for example '15610'
        ref_channel -- (string): reference channel with respect to which the differences are computed
        all_channels -- (bool): If True, plots the histograms for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        for single_run in self.included_runs:
            self.__hist_time_delta_single_run(single_run, ref_channel, all_channels)

            
    # ---- VARIATION OVER RUNS ----
    def __time_delta_run_single_board(self, board, ref_channel):
        """
        Plots the evolution over the runs of the time delta of the channels on the board with respect to the ref_channel.
        
        board -- (string): board considered
        ref_channel -- (string): reference channel with respect to which the differences are computed
        """
        
        # load the Dataframes     
        mean = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma = np.zeros((len(self.included_runs), len(self.numbers)))
        for i, single_run in enumerate(self.included_runs):
            run_time_delta_df = self.__load_stats(single_run, board, ref_channel, 'run') # 4 columns, n_numbers rows
            mean[i,:] = run_time_delta_df["mu"]
            sigma[i,:] = run_time_delta_df["sigma"] 
        
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        
        # Plotting evolution for a single board
        plt.figure()
        for j, number in enumerate(self.numbers):
            plt.errorbar(np.arange(len(self.included_runs)), mean[:,j], yerr=sigma[:,j], label=board+number)
            
        plt.xticks(np.arange(len(self.included_runs)), self.included_runs)
        plt.legend(loc='best')
        plt.title(f'Board {board}, ref {ref_channel}, mean time delta over runs')
        plt.xlabel('Run')
        plt.ylabel('Time delta (ps)')
        plt.show()
    
    # TODO: lancer une exception lorsque liste contient un seul run (pas d'évolution possible)
    def variation_time_delta_run(self, ref_channel, all_channels):
        """
        Plots the evolution over the runs in self.included_runs of the time delta of the channels with respect to the ref_channel on one or all the boards, depending on the value of all_channels.
        
        ref_channel -- (string): reference channel with respect to which the differences are computed
        all_channels -- (bool): If True, we make plots of the time delta evolution with respect to ref_channel for all boards, if False, only plots the time delta evolution for the board of ref_channel.
        """
        if all_channels:
            for board in self.letters:
                self.__time_delta_run_single_board(board, ref_channel)
        else:
            board = ref_channel[0]
            self.__time_delta_run_single_board(board, ref_channel)
            
    
    # ---- STATISTICS OVER RUNS ----
    def __run_statistics_single_run(self, single_run, ref_channel):
        """ 
        Plots mean mu for the time delta with respect to ref_channel of a designated single_run in a colormesh plot.
        The mesh represents all the channels and the color reperesents the time delta.
    
        single_run -- (string or int) The number of a run, for example '15484'
        ref_channel -- (string): reference channel with respect to which the differences are computed
        """
    
        stat_names = ['Mu', 'Mu error', 'Sigma', 'Sigma_error']
        folder =  self.raw_data_folder + str(int(single_run))
        run_name = os.path.basename(os.path.normpath(folder))
        print('Run: ', run_name)
        run_save = self.save_folder + '/Run ' + str(run_name) + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        # TODO: do we also want to plot sigma, mu_err, sigma_err?
        mean = np.zeros((len(self.letters), len(self.numbers)))
        for i, board in enumerate(self.letters):
            run_time_delta_df = self.__load_stats(single_run, board, ref_channel, param='run')
            mean[i,:] = run_time_delta_df["mu"]
        
        plt.figure()
        c = plt.pcolormesh(self.X, self.Y, mean)
        cb = plt.colorbar(c)
        cb.set_label('Mean time delta over channels (ps)')
        plt.title(f'Mean, Run: {run_name}, ref: {ref_channel}')
        plt.show()
        
        plt.savefig(run_save + f'Time Stats Colormesh ref {ref_channel}.pdf', dpi = 300)

        
    def run_statistics(self, ref_channel):
        """ 
        Plots mean mu for the time delta with respect to ref_channel of the runs in self.included_runs in colormesh plots.
        The mesh represents all the channels and the color reperesents the time delta.
    
        ref_channel -- (string): reference channel with respect to which the differences are computed
        """
        for single_run in self.included_runs:
            self.__run_statistics_single_run(single_run, ref_channel)

# -------------------------------------------------------------------------------------
# TODO: delete?

    def variation_statistics(measurement_name, measurement_date, colormesh_max=10, within_board_plot=True):
        """ Plots the mu and sigma and their errors of a measurement over several runs in colormesh plots.

        measurement_name -- (string) Title of the measurement, for example power cycle or temperature
        measurement_date -- (string) Date of the measurement. Originally used to distinguish between series of runs, 
                            but the date will not be unique enough for future measurements.
                            Could and should be replaced by a unique identifier, like an ID for a batch of runs.
        colormesh_max -- (float) The maximum of the scale in the colormesh plot. Lowering this reveals finer differences, but blows out rough ones.
        within_board_plot -- (boolean) Plots the average value of a statistic within the different boards. Still needs refinement as it does not properly
                            handle erroneous channels as of now.
        """

        with h5py.File(self.plot_save_folder + '/' + f'{measurement_name}' + f' {measurement_date}' + ' Stats.h5', 'r') as hf:
            stats_of_stats = hf["stats"][:]

        #print(stats_of_stats[:, :, 1], stats_of_stats.shape)

        plot_titles = ['Mean of Mu', 'Std Dev of Mu', 'Mean of Sigma', 'Std Dev of Sigma']
        variation_save = self.plot_save_folder + '/' + measurement_name + '/' + measurement_date + '/'
        Path(variation_save).mkdir(parents=True, exist_ok=True)

        # Plotting
        for k, ref_channel in enumerate(self.channel_names):
            for i in range(len(stats_of_stats[0,:,0])):

                # Used for skipping means, they are not really interesting without the std dev
                if (i == 0) or (i ==2):
                    continue

                # Visualizing the average value of a statistic within a board. The lowest one (desirable for std dev) is marked red.
                if within_board_plot:
                    if k%5==0:
                        within_board_performances = []
                        for q in range(len(self.letters)):
                            within_board_performances.append(np.mean(stats_of_stats[(q*len(self.numbers)):(q*len(self.numbers))+5, i, k]))
                        within_board_performances = np.array(within_board_performances)
                        print(within_board_performances)
                        plt.figure()
                        barlist = plt.bar(self.letters, height = within_board_performances)
                        barlist[np.argmin(within_board_performances)].set_color('r')
                        plt.title(f'{plot_titles[i]}: Within board performance for reference board {self.letters[k//5]}')
                        plt.show()


                plt.figure()
                stat_data = stats_of_stats[:, i, k].reshape(4,5)
                c = plt.pcolormesh(self.X, self.Y, stat_data, vmin = 0, vmax = 10)
                cb = plt.colorbar(c)
                cb.set_label('Deviation over Temperature (ps)')
                plt.title('Temperature Variation' + '\n' +  f'{plot_titles[i]}, Reference Channel: {ref_channel}')
                plt.savefig(variation_save + f'Variation Stats Colormesh {plot_titles[i]} Ref Channel {self.__to_channel_converter(k)}.pdf', dpi = 300)      
                plt.show()