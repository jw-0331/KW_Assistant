#!/usr/bin/env python
# coding: utf-8

# In[55]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import import_ipynb
# Import local Libraries
#from ECG_feature_extractor import Features
from ECG_feature_extract import Features


# Configure Notebook
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')




# sampling rate(frequency)  (Hz)
fs = 1000 

waveform_path = os.path.join(os.path.dirname(os.getcwd()), 'DATA', 'ECG')  #읽어올 .mat 파일 위치
feature_path = os.path.join(os.path.dirname(os.getcwd()), 'ECG_feature', 'Feature')

# Instantiate
ecg_features = Features(file_path=waveform_path, fs=fs, feature_groups=['full_waveform_features'])

# Calculate ECG features
ecg_features.extract_features(
    filter_bandwidth=[0.5, 35], n_signals=None, show=True, normalize=True,
    template_before=0.2, template_after=0.4
)


# In[56]:


features = ecg_features.get_features()

features


# In[57]:


# Save features 
features.to_csv(os.path.join(feature_path, 'features.csv'), index=False)

