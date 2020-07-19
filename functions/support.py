import numpy as np
from numba import jit
import mne
import pandas as pd

tcp01 = {
    "FP1-F7": ["EEG FP1-REF", "EEG F7-REF"],
    "F7-T3": ["EEG F7-REF", "EEG T3-REF"],
    "T3-T5": ["EEG T3-REF", "EEG T5-REF"],
    "T5-O1": ["EEG T5-REF", "EEG O1-REF"],
    "FP2-F8": ["EEG FP2-REF", "EEG F8-REF"],
    "F8-T4": ["EEG F8-REF", "EEG T4-REF"],
    "T4-T6": ["EEG T4-REF", "EEG T6-REF"],
    "T6-O2": ["EEG T6-REF", "EEG O2-REF"],
    "A1-T3": ["EEG A1-REF", "EEG T3-REF"],
    "T3-C3": ["EEG T3-REF", "EEG C3-REF"],
    "C3-CZ": ["EEG C3-REF", "EEG CZ-REF"],
    "CZ-C4": ["EEG CZ-REF", "EEG C4-REF"],
    "C4-T4": ["EEG C4-REF", "EEG T4-REF"],
    "T4-A2": ["EEG T4-REF", "EEG A2-REF"],
    "FP1-F3": ["EEG FP1-REF", "EEG F3-REF"],
    "F3-C3": ["EEG F3-REF", "EEG C3-REF"],
    "C3-P3": ["EEG C3-REF", "EEG P3-REF"],
    "P3-O1": ["EEG P3-REF", "EEG O1-REF"],
    "FP2-F4": ["EEG FP2-REF", "EEG F4-REF"],
    "F4-C4": ["EEG F4-REF", "EEG C4-REF"],
    "C4-P4": ["EEG C4-REF", "EEG P4-REF"],
    "P4-O2": ["EEG P4-REF", "EEG O2-REF"]}


tcp02 = {
    "FP1-F7": ["EEG FP1-LE", "EEG F7-LE"],
    "F7-T3": ["EEG F7-LE", "EEG T3-LE"],
    "T3-T5": ["EEG T3-LE", "EEG T5-LE"],
    "T5-O1": ["EEG T5-LE", "EEG O1-LE"],
    "FP2-F8": ["EEG FP2-LE", "EEG F8-LE"],
    "F8-T4": ["EEG F8-LE", "EEG T4-LE"],
    "T4-T6": ["EEG T4-LE", "EEG T6-LE"],
    "T6-O2": ["EEG T6-LE", "EEG O2-LE"],
    "A1-T3": ["EEG A1-LE", "EEG T3-LE"],
    "T3-C3": ["EEG T3-LE", "EEG C3-LE"],
    "C3-CZ": ["EEG C3-LE", "EEG CZ-LE"],
    "CZ-C4": ["EEG CZ-LE", "EEG C4-LE"],
    "C4-T4": ["EEG C4-LE", "EEG T4-LE"],
    "T4-A2": ["EEG T4-LE", "EEG A2-LE"],
    "FP1-F3": ["EEG FP1-LE", "EEG F3-LE"],
    "F3-C3": ["EEG F3-LE", "EEG C3-LE"],
    "C3-P3": ["EEG C3-LE", "EEG P3-LE"],
    "P3-O1": ["EEG P3-LE", "EEG O1-LE"],
    "FP2-F4": ["EEG FP2-LE", "EEG F4-LE"],
    "F4-C4": ["EEG F4-LE", "EEG C4-LE"],
    "C4-P4": ["EEG C4-LE", "EEG P4-LE"],
    "P4-O2": ["EEG P4-LE", " EEG O2-LE"]}

tcp03 = {
    "FP1-F7": ["EEG FP1-REF", "EEG F7-REF"],
    "F7-T3": ["EEG F7-REF", "EEG T3-REF"],
    "T3-T5": ["EEG T3-REF", "EEG T5-REF"],
    "T5-O1": ["EEG T5-REF", "EEG O1-REF"],
    "FP2-F8": ["EEG FP2-REF", "EEG F8-REF"],
    "F8-T4": ["EEG F8-REF", "EEG T4-REF"],
    "T4-T6": ["EEG T4-REF", "EEG T6-REF"],
    "T6-O2": ["EEG T6-REF", "EEG O2-REF"],
    "T3-C3": ["EEG T3-REF", "EEG C3-REF"],
    "C3-CZ": ["EEG C3-REF", "EEG CZ-REF"],
    "CZ-C4": ["EEG CZ-REF", "EEG C4-REF"],
    "C4-T4": ["EEG C4-REF", "EEG T4-REF"],
    "FP1-F3": ["EEG FP1-REF", "EEG F3-REF"],
    "F3-C3": ["EEG F3-REF", "EEG C3-REF"],
    "C3-P3": ["EEG C3-REF", "EEG P3-REF"],
    "P3-O1": ["EEG P3-REF", "EEG O1-REF"],
    "FP2-F4": ["EEG FP2-REF", "EEG F4-REF"],
    "F4-C4": ["EEG F4-REF", "EEG C4-REF"],
    "C4-P4": ["EEG C4-REF", "EEG P4-REF"],
    "P4-O2": ["EEG P4-REF", "EEG O2-REF"]
}

tcp01 = pd.DataFrame(tcp01)
tcp02 = pd.DataFrame(tcp02)
tcp03 = pd.DataFrame(tcp03)



def aux(x, func):
    fs = 256
    m = np.zeros(x.shape[0]//fs)
    for i in range(len(m)):
        m[i] = func(x[i*fs:(i+1)*fs])
    return m


def read_edf(path):
    raw = mne.io.read_raw_edf(path, preload=True).to_data_frame()
    return raw.T


def orgMontage(x, tipo):
    
    m = []
    
    if tipo == '01_tcp_ar':
        use = tcp01
    elif tipo == '02_tcp_le':
        use = tcp02
    else:
        use = tcp03

    for i in range(len(use.columns)):
        m.append(x.loc[use.iloc[0,i],:] - x.loc[use.iloc[1,i],:])
    return pd.DataFrame(data=np.array(m),index=use.columns) 