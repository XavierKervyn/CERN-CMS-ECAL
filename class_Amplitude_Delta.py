""" Imports """

from class_ECAL import *


""" 2nd Child Class definition """     
        
class Amplitude_Delta(ECAL):
    """
    This class is for the analysis of the amplitude resolution of the detector
    """
    def __init__(self, included_runs, letters, # TODO: check if h5 file generated for the runs
                 save_folder = save_folder_global, raw_data_folder = raw_data_folder_global,
                 plot_save_folder = plot_save_folder_global):
         super().__init__(included_runs, letters, save_folder, raw_data_folder, plot_save_folder)
    

    
    def __to_channel_converter(self, channel_number):
        """ Converts the channel number to the appropriate Channel. For example 7 -> 'B3'. """
        board_counter = 0
        while channel_number > 4:
            board_counter += 1
            channel_number -= 5
        return f'{self.letters[board_counter]}{self.numbers[channel_number]}'

    def __compute_amplitude_delta(self, amp, board, ref_channel):
        # TODO: change docstring
        """ Computes the amplitude difference (delta) for a given reference channel to all the other channels. 
        Also returns the mu and sigma statistics and their errors."""
        
        amp_pd = pd.DataFrame(amp, columns=self.channel_names)
        n_numbers = len(self.numbers)

        amp_delta_pd = pd.DataFrame()

        slicing = [channel for channel in self.channel_names if channel[0]==board]
        for i, channel in enumerate(slicing):
            if channel == ref_channel:
                continue

            reference_amp = amp_pd[ref_channel]
            curr_amp = amp_pd[channel]
            amp_delta = curr_amp - reference_amp # TODO: check sign

            # Save amplitude deltas for later analysis
           
            amp_delta_pd[f'{channel}'] = amp_delta
        return amp_delta_pd
        
    def __generate_stats(self, single_run, board, ref_channel, param='run', plot=False):
        # TODO: change docstring
        """ 
        Computes the amplitude deltas for a single run. 
        The splits are merged into a single big run number and the amplitude deltas are saved in an h5 file.

        singe_run -- (string or int) the number of a run, for example '15610'
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
        amp = h2['amp_max']
        amp_pd = pd.DataFrame(amp, columns=self.channel_names)
        
        # column header for the Dataframes
        col_list = len(self.numbers)*[board]; col_list = [x + y for x,y in zip(col_list, self.numbers)]
        
        if param=='spill':
            # Computation with merged data: retrieve the spill number
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
                
                amp_delta_pd = self.__compute_amplitude_delta(aspill_pd_temp, board, ref_channel)
                
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
                        hist, bin_edges, _ = plt.hist(amp_delta_pd[channel], bins = 1500, label="Amplitude Histogram")
                    else:
                        hist, bin_edges = np.histogram(amp_delta_pd[channel], bins = 1500)

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
                        plt.xlabel('Amplitude delta (??)')
                        plt.ylabel('Occurence (a.u.)')
                        plt.legend(loc='best')

                        plt.show()
                    
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
            spill_amp_mean_df.to_csv(self.save_folder + f'/Run {single_run}' + f'/Spill mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
            spill_amp_mean_err_df.to_csv(self.save_folder + f'/Run {single_run}' + f'/Spill error mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
            spill_amp_sigma_df.to_csv(self.save_folder + f'/Run {single_run}' + f'/Spill sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
            spill_amp_sigma_err_df.to_csv(self.save_folder + f'/Run {single_run}' + f'/Spill error sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
                
        elif param=='run':
            amp_delta_pd = self.__compute_amplitude_delta(amp_pd, board, ref_channel)
            
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
                    hist, bin_edges, _ = plt.hist(amp_delta_pd[channel], bins = 1500, label="Amplitude Histogram")
                else:
                    hist, bin_edges = np.histogram(amp_delta_pd[channel], bins = 1500)

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
                        plt.xlabel('Amplitude delta (??)')
                        plt.ylabel('Occurence (a.u.)')
                        plt.legend(loc='best')

                        plt.show()

            # convert the arrays into a single Dataframe
            run_amp_delta_df = pd.DataFrame({'mu':mu_arr, 'mu error':mu_error_arr, 'sigma': sigma_arr, 'sigma error': sigma_error_arr})

            # save it in a .csv file
            run_amp_delta_df.to_csv(self.save_folder + f'/Run {single_run}' + f'/Run amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
        
        else: # TODO: throw exception
            print('wrong parameter, either spill or run')
            
    def __load_stats(self, single_run, board, ref_channel, param):
        # TODO: docstring
        
        try: # check if the file exits
            # loading the file and returning it
            if param=='spill': # returns a tuple with the 4 files
                return (pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Spill mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Spill error mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Spill sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Spill error sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv'))
            elif param=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Run amplitude delta run {single_run} board {board} ref {ref_channel}.csv')     
        except FileNotFoundError:
            print('File not found, generating .csv')
            self.__generate_stats(single_run, board, ref_channel, param) # generating the statistics file
        
            # loading the file and returning it
            if param=='spill': # returns a tuple with the 4 files
                return (pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Spill mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Spill error mean amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Spill sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv'),
                    pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Spill error sigma amplitude delta run {single_run} board {board} ref {ref_channel}.csv'))

            elif param=='run':
                return pd.read_csv(self.save_folder + f'/Run {single_run}' + f'/Run amplitude delta run {single_run} board {board} ref {ref_channel}.csv')
        except:
            raise Exception('Could not load nor generate .csv file')
        
            
    # ------------------------------------------------------------------------------------------------------------------------------
    # SPILLS
    
    def __amplitude_delta_spill_single_board(self, single_run, board, ref_channel): 
        # TODO: docstring
        
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
        plt.title(f'Run {single_run}, board {board}, ref {ref_channel}, mean amplitude delta over spills')
        plt.xlabel('Spill')
        plt.ylabel('Amplitude delta (ps)')
        plt.show()

    def __amplitude_delta_spill_single_run(self, single_run, ref_channel, all_channels):
        # TODO: docstring
        if all_channels:
            for board in self.letters:
                self.__amplitude_delta_spill_single_board(single_run, board, ref_channel)
        else:
            board = ref_channel[0]
            self.__amplitude_delta_spill_single_board(single_run, board, ref_channel)
    
    def variation_amplitude_delta_spill(self, ref_channel, all_channels):
        # TODO: docstring
        
        for single_run in self.included_runs:
            self.__amplitude_delta_spill_single_run(single_run, ref_channel, all_channels)

    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS
    
    def __hist_amplitude_delta_single_board(self, single_run, board, ref_channel):
        # TODO: docstring
        self.__generate_stats(single_run, board, ref_channel, param='run', plot=True)
        
    def __hist_amplitude_delta_single_run(self, single_run, ref_channel, all_channels):
        # TODO: docstring
        if all_channels:
            for board in self.letters:
                self.__hist_amplitude_delta_single_board(single_run, board, ref_channel)
        else:
            board = ref_channel[0]
            self.__hist_amplitude_delta_single_board(single_run, board, ref_channel)
            
    def hist_amplitude_delta(self, ref_channel, all_channels):
        """
        Plots the histogram for every single run in the included_runs list (parent attribute)
        """
        for single_run in self.included_runs:
            self.__hist_amplitude_delta_single_run(single_run, ref_channel, all_channels)

    def __amplitude_delta_run_single_board(self, board, ref_channel):
        # TODO: docstring
        
        # load the Dataframes     
        mean = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma = np.zeros((len(self.included_runs), len(self.numbers)))
        for i, single_run in enumerate(self.included_runs):
            run_amplitude_delta_df = self.__load_stats(single_run, board, ref_channel, 'run') # 4 columns, n_numbers rows
            mean[i,:] = run_amplitude_delta_df["mu"]
            sigma[i,:] = run_amplitude_delta_df["sigma"] 
        
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        
        # Plotting evolution for a single board
        plt.figure()
        for j, number in enumerate(self.numbers):
            plt.errorbar(np.arange(len(self.included_runs)), mean[:,j], yerr=sigma[:,j], label=board+number)
            
        plt.xticks(np.arange(len(self.included_runs)), self.included_runs)
        plt.legend(loc='best')
        plt.title(f'Board {board}, ref {ref_channel}, mean amplitude delta over runs')
        plt.xlabel('Run')
        plt.ylabel('Amplitude delta (??)')
        plt.show()
    
    # TODO: lancer une exception lorsque liste contient un seul run (pas d'évolution possible)
    def variation_amplitude_delta_run(self, ref_channel, all_channels):
        # TODO: correct docstring
        """
        Plots the evolution of the mu and sigma statistics and their errors over a number of runs.
    
        measurement_name -- (string) Title of the measurement, for example power cycle or temperature
        measurement_date -- (string) Date of the measurement. Originally used to distinguish between series of runs, 
                            but the date will not be unique enough for future measurements.
                            Could and should be replaced by a unique identifier, like an ID for a batch of runs.
        included_runs -- (list of strings or ints) List of all the runs to include in our variation plot.
        specific_ref_channel -- (string) If one wants to test the function only for a specific channel, use that channel here, 
        for example 'B3'
        """
        if all_channels:
            for board in self.letters:
                self.__amplitude_delta_run_single_board(board, ref_channel)
        else:
            board = ref_channel[0]
            self.__amplitude_delta_run_single_board(board, ref_channel)
    
    def __run_statistics_single_run(self, single_run, ref_channel):
        """ 
        Plots mu and sigma as well as their errors for the amplitude of a designated single run in a colormesh plot.
        One has to have run the run_amplitude_computation function on the designated run first before using this function.
    
        single_run -- (string or int) The number of a run, for example '15484'
        save_folder -- (string) Folder where the computed data should be stored
        raw_data_folder -- (string) Folder where the raw experiment data is located
        skip_mu == (boolean) If one is not interested in mu, one can skip plotting it
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
            run_amplitude_delta_df = self.__load_stats(single_run, board, ref_channel, param='run')
            mean[i,:] = run_amplitude_delta_df["mu"]
        
        plt.figure()
        c = plt.pcolormesh(self.X, self.Y, mean)
        cb = plt.colorbar(c)
        cb.set_label('Mean amplitude delta over channels (ps)')
        plt.title(f'Mean, Run: {run_name}, ref: {ref_channel}')
        plt.show()
        
        plt.savefig(run_save + f'Amplitude Stats Colormesh ref {ref_channel}.pdf', dpi = 300)

        
    def run_statistics(self, ref_channel):
        """
        Plots the colormesh map for every single run in the included_runs list (parent attribute)
        """
        for single_run in self.included_runs:
            self.__run_statistics_single_run(single_run, ref_channel)

# -------------------------------------------------------------------------------------
# TODO: delete

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