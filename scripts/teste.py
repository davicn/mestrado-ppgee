import numpy as np
from os.path import expanduser
import sys
sys.path.append('./')
from functions.support import edfArray

home = expanduser('~')+'/edf/'

c = data = np.load('data/train_seizure.npy')

v = edfArray(home+c[0, 0].replace('.tse', '.edf'), 'EKG')
print(v)