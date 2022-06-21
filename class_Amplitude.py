""" Imports """

from class_ECAL import *


""" 1st Child Class definition """

class Amplitude(ECAL):
    """
    This class is for the analysis of the amplitudes
    """
    def __init__(self, included_runs, letters, # TODO: check if h5 file generated for the runs
                 save_folder = save_folder_global, raw_data_folder = raw_data_folder_global,
                 plot_save_folder = plot_save_folder_global):
        super().__init__(included_runs, letters, save_folder, raw_data_folder, plot_save_folder)
        
    # ------------------------------------------------------------------------------------------------------------------------------
    # GENERAL
    
    def __generate_stats(self, single_run, board, param='run', plot=False):
        # TODO: docstring
        
        folder =  self.raw_data_folder + str(int(single_run))

        # Computation with merged data: retrieve the amplitude
        folder =  self.raw_data_folder + str(int(single_run))
        h2 = uproot.concatenate({folder + '/*.root' : 'digi'}, allow_missing = True)
        run_name = os.path.basename(os.path.normpath(folder)) # creating folder to save h5 file
        # TODO: delete print or add verbose boolean parameter?
        print('Run: ', run_name)
        run_save = self.save_folder + '/Run ' + run_name + '/' + self.split_name + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True) # folder created
        
        # retrieve only the channels for the given board
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        amp = h2['amp_max'] # retrieve the amplitude
        amp_pd = pd.DataFrame(amp, columns=self.channel_names)[slicing]
        
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

                # 'empty' arrays to store the statistics of each channel
                mu_arr = np.zeros(len(self.numbers))
                mu_error_arr = np.zeros(len(self.numbers))
                sigma_arr = np.zeros(len(self.numbers))
                sigma_error_arr = np.zeros(len(self.numbers))

                for i, channel in enumerate(slicing):
                    border_size = 2000 # TODO: is not useful?
                    
                    if plot:
                        plt.figure()
                        hist, bin_edges, _ = plt.hist(aspill_pd_temp[channel], bins = 1500, label="Amplitude Histogram")
                    else:
                        hist, bin_edges = np.histogram(aspill_pd_temp[channel], bins = 1500)

                    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  

                    # fitting process
                    guess = [np.max(hist), bin_centers[np.argmax(hist)], 100]
                    coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess)
                    mu = coeff[1]
                    mu_error = covar[1,1]
                    sigma = coeff[2]
                    sigma_error = covar[2,2]
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

                        plt.title(f'Run: {run_name}, Channel: {board+self.numbers[i]}, Spill {spill}')
                        plt.xlabel('Amplitude (??)')
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
            spill_amp_mean_df.to_csv(self.save_folder + f'/Spill mean amplitude run {single_run} board {board}.csv')
            spill_amp_mean_err_df.to_csv(self.save_folder + f'/Spill error mean amplitude run {single_run} board {board}.csv')
            spill_amp_sigma_df.to_csv(self.save_folder + f'/Spill sigma amplitude run {single_run} board {board}.csv')
            spill_amp_sigma_err_df.to_csv(self.save_folder + f'/Spill error sigma amplitude run {single_run} board {board}.csv')
        
        elif param=='run':
            # 'empty' arrays to store the statistics of each channel
            mu_arr = np.zeros(len(self.numbers))
            mu_error_arr = np.zeros(len(self.numbers))
            sigma_arr = np.zeros(len(self.numbers))
            sigma_error_arr = np.zeros(len(self.numbers))

            for i, channel in enumerate(slicing):
                border_size = 2000 # TODO: is not useful?

                if plot:
                    plt.figure()
                    hist, bin_edges, _ = plt.hist(amp_pd[channel], bins = 1500, label="Amplitude Histogram")
                else:
                    hist, bin_edges = np.histogram(amp_pd[channel], bins = 1500)

                bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  

                # fitting process
                guess = [np.max(hist), bin_centers[np.argmax(hist)], 100]
                coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess)
                mu = coeff[1]
                mu_error = covar[1,1]
                sigma = coeff[2]
                sigma_error = covar[2,2]
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

                        plt.title(f'Run: {run_name}, Channel: {board+self.numbers[i]}')
                        plt.xlabel('Amplitude (??)')
                        plt.ylabel('Occurence (a.u.)')
                        plt.legend(loc='best')

                        plt.show()

            # convert the arrays into a single Dataframe
            run_amp_df = pd.DataFrame({'mu':mu_arr, 'mu error':mu_error_arr, 'sigma': sigma_arr, 'sigma error': sigma_error_arr})

            # save it in a .csv file
            run_amp_df.to_csv(self.save_folder + f'/Run amplitude run {single_run} board {board}.csv')
        
        else: # TODO: throw exception
            print('wrong parameter, either spill or run')
    
    def __load_stats(self, single_run, board, param):
        # TODO: docstring
        # TODO: throw exception or generate file if file does not exist
        
        # TODO: remove when exception implemented
        self.__generate_stats(single_run, board, param) # generating the statistics file
        
        # loading the file and returning it
        if param=='spill': # returns a tuple with the 4 files
            return (pd.read_csv(self.save_folder + f'/Spill mean amplitude run {single_run} board {board}.csv'),
                pd.read_csv(self.save_folder + f'/Spill error mean amplitude run {single_run} board {board}.csv'),
                pd.read_csv(self.save_folder + f'/Spill sigma amplitude run {single_run} board {board}.csv'),
                pd.read_csv(self.save_folder + f'/Spill error sigma amplitude run {single_run} board {board}.csv'))
        
        elif param=='run':
            return pd.read_csv(self.save_folder + f'/Run amplitude run {single_run} board {board}.csv')
        
        else:
            # TODO: throw exception
            print('wrong parameter, either spill/run')
            
    # ------------------------------------------------------------------------------------------------------------------------------
    # SPILLS
    
    # TODO should be private
    def __amplitude_spill_single_board(self, single_run, board): 
        # TODO: docstring
        
        # load the Dataframes
        mean, mean_err, sigma, sigma_err = self.__load_stats(single_run, board, 'spill')
        num_spills = mean.shape[0] # number of spills in the single run
        
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        
        # Plotting evolution for a single board
        plt.figure()
        for i, number in enumerate(self.numbers):
            plt.errorbar(np.arange(num_spills), mean[slicing[i]], yerr=sigma[slicing[i]], label=board+number)
            
        plt.xticks(np.arange(num_spills), 1+np.arange(num_spills)) # TODO: check
        plt.legend(loc='best')
        plt.title(f'Run {single_run}, board {board}, mean amplitude over spills')
        plt.xlabel('Spill')
        plt.ylabel('Amplitude (??)')
        plt.show()
    
    
    # TODO: should be private
    def __amplitude_spill_single_run(self, single_run):
        # TODO: docstring
        for board in self.letters:
            self.__amplitude_spill_single_board(single_run, board)
    
    
    def variation_amplitude_spill(self):
        # TODO: docstring
        
        for single_run in self.included_runs:
            self.__amplitude_spill_single_run(single_run)  

    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS
    
    def __hist_amplitude_single_board(self, single_run, board):
        # TODO: docstring
        self.__generate_stats(single_run, board, 'run', True)
        
    # TODO: should be private / shouldn't be accessed from outside
    def __hist_amplitude_single_run(self, single_run):
        # TODO: docstring
        for board in self.letters:
            self.__hist_amplitude_single_board(single_run, board)
        
    # TODO: faire en sorte qu'une liste appelle mutliple et un entier seul appelle single
    def hist_amplitude(self):
        """
        Plots the histogram for every single run in the included_runs list (parent attribute)
        """
        for single_run in self.included_runs:
            self.__hist_amplitude_single_run(single_run)

    def __amplitude_run_single_board(self, board):
        # TODO: docstring
        
        # load the Dataframes     
        mean = np.zeros((len(self.included_runs), len(self.numbers)))
        sigma = np.zeros((len(self.included_runs), len(self.numbers)))
        for i, single_run in enumerate(self.included_runs):
            run_amp_df = self.__load_stats(single_run, board, 'run') # 4 columns, n_numbers rows
            mean[i,:] = run_amp_df["mu"]
            sigma[i,:] = run_amp_df["sigma"] 
        
        slicing = [channel for channel in self.channel_names if channel[0] == board]
        
        # Plotting evolution for a single board
        plt.figure()
        for j, number in enumerate(self.numbers):
            plt.errorbar(np.arange(len(self.included_runs)), mean[:,j], yerr=sigma[:,j], label=board+number)
            
        plt.xticks(np.arange(len(self.included_runs)), self.included_runs)
        plt.legend(loc='best')
        plt.title(f'Board {board}, mean amplitude over runs')
        plt.xlabel('Run')
        plt.ylabel('Amplitude (??)')
        plt.show()
    
    # TODO: add specific channel
    # TODO: lancer une exception lorsque liste contient un seul run (pas d'évolution possible)
    def variation_amplitude_run(self):
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
        for board in self.letters:
            self.__amplitude_run_single_board(board)
    
    def __run_statistics_single_run(self, single_run):
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
        run_save = self.save_folder + '/Run ' + str(run_name) + '/' + self.split_name + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        # TODO: do we also want to plot sigma, mu_err, sigma_err?
        mean = np.zeros((len(self.letters), len(self.numbers)))
        for i, board in enumerate(self.letters):
            run_amp_df = self.__load_stats(single_run, board, 'run')
            mean[i,:] = run_amp_df["mu"]
        #with h5py.File(run_save + 'Statistics Split ' + self.split_name + '.h5', 'r') as hf:
        #    statistics = hf[f"stats"][:]

        plt.figure()
        c = plt.pcolormesh(self.X, self.Y, mean)
        cb = plt.colorbar(c)
        cb.set_label('Max amplitude over Channels (??)')
        plt.title(f'Mean, Run: {run_name}')
        plt.show()
        
        plt.savefig(run_save + f'Stats Colormesh.pdf', dpi = 300)

        
    def run_statistics(self):
        """
        Plots the colormesh map for every single run in the included_runs list (parent attribute)
        """
        for single_run in self.included_runs:
            self.__run_statistics_single_run(single_run)
