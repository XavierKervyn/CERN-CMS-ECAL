# imports
from ecalautoanalysis import *
import numpy as np

# included runs

#included_runs = [15680]
#included_runs = list(np.arange(15728, 15773+1))

# list of boards connected

letters = ['A'] 

# instances

a = Amplitude(included_runs, letters)

# extra arguments

ref_channel_ = 'C2'
all_channels_ = False
variation_ = 'spill'
spill_i_ = 3

# methods (uncomment)

#a.spill_variation()
#a.hist()
#a.run_variation('run 4 boards 29 June 2022')
#a.run_colormesh()
a.resolution('run 4 boards 29 June 2022')

