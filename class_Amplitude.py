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
    
    def generate_stats(self, single_run, board, param, plot=False):
        # TODO: docstring
        
        folder =  self.raw_data_folder + str(int(single_run))

        # Computation with merged data: retrieve the amplitude
        folder =  self.raw_data_folder + str(int(single_run))
        h2 = uproot.concatenate({folder + '/*.root' : 'digi'}, allow_missing = True)
        run_name = os.path.basename(os.path.normpath(folder)) # creating folder to save h5 file
        print('Run: ', run_name, ' Split: ', self.split_name)
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

                        plt.title(f'Run: {run_name}, Channel: {self.channel_names[i]}, Spill {spill}')
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

            # convert the arrays into a single Dataframe
            run_amp_df = pd.DataFrame({'mu':mu_arr, 'mu error':mu_error_arr, 'sigma': sigma_arr, 'sigma error': sigma_error_arr})

            # save it in a .csv file
            run_amp_df.to_csv(self.save_folder + f'/Run amplitude run {single_run} board {board}.csv')
        
        else: # TODO: throw exception
            print('wrong parameter, either spill or run')
    
    def load_stats(self, single_run, board, param):
        # TODO: docstring
        # TODO: throw exception or generate file if file does not exist
        
        # TODO: remove when exception implemented
        self.generate_stats(single_run, board, param) # generating the statistics file
        
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
    def amplitude_spill_single_board(self, board, single_run): 
        # TODO: docstring
        
        # load the Dataframes
        mean, mean_err, sigma, sigma_err = self.load_stats(single_run, board, 'spill')
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
    def amplitude_spill_single_run(self, single_run):
        # TODO: docstring
        for board in self.letters:
            self.amplitude_spill_single_board(board, single_run)
    
    
    def variation_amplitude_spill(self):
        # TODO: docstring
        
        for single_run in self.included_runs:
            self.amplitude_spill_single_run(single_run)  

    # ------------------------------------------------------------------------------------------------------------------------------
    # RUNS
    
    # TODO: should be private / shouldn't be accessed from outside
    def run_amplitude_single_run(self, run_number):
        """ 
        Computes the amplitude for a single run. 
        The splits are merged into a single big run number and the time deltas are saved in an h5 file.
    
        run_number -- (string or int) the number of a run, for example '15610'
        """
    
        # Computation with merged data
        folder =  self.raw_data_folder + str(int(run_number))
        h = uproot.concatenate({folder + '/*.root' : 'digi'}, allow_missing = True)
    
        run_name = os.path.basename(os.path.normpath(folder))
        print('Run: ', run_name, ' Split: ', self.split_name)
        run_save = self.save_folder + '/Run ' + run_name + '/' + self.split_name + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        amp = h['amp_max'] # retrieve the amplitude
        amp_pd = pd.DataFrame(amp)
    
        # 'empty' arrays to store the statistics of each channel
        mu_arr = np.zeros(len(self.channel_names))
        mu_error_arr = np.zeros(len(self.channel_names))
        sigma_arr = np.zeros(len(self.channel_names))
        sigma_error_arr = np.zeros(len(self.channel_names))
    
        for i in range(len(self.channel_names)):
            plt.figure()
            border_size = 2000

            hist, bin_edges, _ = plt.hist(amp_pd[i], bins = 1500, label= 'Amplitude Histogram')
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)  
        
            # fitting process
            guess = [np.max(hist), bin_centers[np.argmax(hist)], 3000]
            coeff, covar = curve_fit(gaussian, bin_centers, hist, p0=guess)
            mu = coeff[1]
            mu_error = covar[1,1]
            sigma = coeff[2]
            sigma_error = covar[2,2]
            mu_arr[i] = mu
            mu_error_arr[i] = mu_error
            sigma_arr[i] = sigma
            sigma_error_arr[i] = sigma_error
        
            # TODO: put correct units for amplitude
            # plotting the histogram with a gaussian fit, the mean and the standard deviation
            plt.plot(bin_centers, gaussian(bin_centers, *coeff), label='Gaussian Fit')
            plt.axvline(mu, label = f'Mean: {np.around(mu, decimals = 1)} ??', color = 'red')
            sigma_color = 'pink'
            plt.axvline(mu + sigma, label = f'Std Dev: {np.around(sigma, decimals = 1)} ??', color = sigma_color)
            plt.axvline(mu - sigma, color = sigma_color)
        
            plt.title(f'Run: {run_name}, Channel: {self.channel_names[i]}')
            plt.xlabel('Amplitude (??)')
            plt.ylabel('Occurence (a.u.)')
            plt.legend(loc='best')
            
            plt.show()
        
        # saving the statistics in a file for later use, eg. in statistics_plot
        statistics = np.hstack((mu_arr.reshape(-1,1), mu_error_arr.reshape(-1,1), 
                                sigma_arr.reshape(-1,1), sigma_error_arr.reshape(-1,1)))
        
        with h5py.File(run_save + 'Statistics Split ' + self.split_name + '.h5', 'w') as hf:
            hf.create_dataset("stats",  data=statistics)
        

    # TODO: faire en sorte qu'une liste appelle mutliple et un entier seul appelle single
    def run_amplitude(self):
        """
        Plots the histogram for every single run in the included_runs list (parent attribute)
        """
        for single_run in self.included_runs:
            self.run_amplitude_single_run(single_run)

    
    def run_statistics_single_run(self, run_number):
        """ 
        Plots mu and sigma as well as their errors for the amplitude of a designated single run in a colormesh plot.
        One has to have run the run_amplitude_computation function on the designated run first before using this function.
    
        run_number -- (string or int) The number of a run, for example '15484'
        save_folder -- (string) Folder where the computed data should be stored
        raw_data_folder -- (string) Folder where the raw experiment data is located
        skip_mu == (boolean) If one is not interested in mu, one can skip plotting it
        """
    
        stat_names = ['Mu', 'Mu error', 'Sigma', 'Sigma_error']
        folder =  self.raw_data_folder + str(int(run_number))
        run_name = os.path.basename(os.path.normpath(folder))
        print('Run: ', run_name, ' Split: ', self.split_name)
        run_save = self.save_folder + '/Run ' + str(run_name) + '/' + self.split_name + '/'
        Path(run_save).mkdir(parents=True, exist_ok=True)

        with h5py.File(run_save + 'Statistics Split ' + self.split_name + '.h5', 'r') as hf:
            statistics = hf[f"stats"][:]

        for i in range(len(statistics[0,:])):
            plt.figure()
            stat_data = statistics[:,i].reshape(len(self.letters), len(self.numbers))
            c = plt.pcolormesh(self.X, self.Y, stat_data)
            cb = plt.colorbar(c)
            cb.set_label('Max amplitude over Channels (??)')
            plt.title(f'{stat_names[i]}, Run: {run_name}, Split: {self.split_name}')
            plt.show()
        
        plt.savefig(run_save + f'Stats Colormesh.pdf', dpi = 300)

        
    def run_statistics(self):
        """
        Plots the colormesh map for every single run in the included_runs list (parent attribute)
        """
        for single_run in self.included_runs:
            self.run_statistics_single_run(single_run)
        print('-- Colormesh plot(s) finished --')
        
    
    # TODO: write similar function to analyse the evolution(spills) for a single run
    # TODO: test with a specific reference channel
    # TODO: lancer une exception lorsque liste contient un seul run (pas d'évolution possible)
    def variation_plot_runs(self, specific_ref_channel='all'):
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
    
        for k, run_name in enumerate(self.included_runs):
            run_save = self.save_folder + '/Run ' + str(run_name) + '/' + self.split_name + '/'
            if k==0: # create the stacked_average_stats first
                with h5py.File(run_save + 'Statistics Split ' + self.split_name + '.h5', 'r') as hf:
                    stacked_average_stats = hf[f"stats"][:]
            else: # then stack for each run included
                with h5py.File(run_save + 'Statistics Split ' + self.split_name + '.h5', 'r') as hf:
                    stacked_average_stats = np.dstack((stacked_average_stats, hf[f"stats"][:]))            
            # stacked_average_stats has shape (n_channels, n_stats, n_runs)
    
        # TODO: exception pour ce genre de pb
        # now select a particular channel if given in specific_ref_channel
        if specific_ref_channel != 'all':
            if (specific_ref_channel not in self.channel_names):
                print('specified channel given not in list, no particular channel selected')
            else:
                index = np.where(channel_names == specific_ref_channel)
                stacked_average_stats = stacked_average_stats[index[0][0],:,:]
    
        for i, board in enumerate(self.letters): # one plot per board letter 'A', ...
            plt.figure()
        
            # one line plotted per channel 1, ..., 5 in each board
            for j, nb in enumerate(self.numbers):
                plt.errorbar(np.arange(len(self.included_runs)), stacked_average_stats[(5*i)+j,0,:],
                             yerr=stacked_average_stats[(5*i)+j,2,:], label=self.channel_names[(5*i)+j])
                plt.xticks(np.arange(len(self.included_runs)), self.included_runs)
            
            plt.legend(loc='best')
            plt.title(f'Board {board}, mean amplitude over runs')
            plt.xlabel('Run')
            plt.ylabel('Amplitude (??)')
            plt.show()