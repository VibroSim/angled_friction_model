import numpy as np
from matplotlib import pyplot as pl
import pickle
import copy
import os 
import os.path
import sys
import scipy
import scipy.optimize
from VibroSim_Simulator.function_as_script import scriptify
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline

pickle_filename = "/tmp/afmdebug93383_15.pickle"
pickle_fh=open(pickle_filename,"rb")
vars = pickle.load(pickle_fh)

globals().update(vars)
assert(crack_model_class=="Tada_ModeI_CircularCrack_along_midline")
scp.crack_model=Tada_ModeI_CircularCrack_along_midline(E,nu)
