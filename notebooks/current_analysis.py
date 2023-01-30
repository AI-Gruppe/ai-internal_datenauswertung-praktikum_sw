# %% Import 3rd Party
import os

import sys
import logging
import numpy as np
import pandas as pd
import IPython.display as ipd

import librosa
import librosa.display


# %% Is Notebook check
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

currentdir = os.path.dirname(os.path.realpath("__file__"))
parentdir = os.path.dirname(currentdir)
if isnotebook():
    sys.path.append(parentdir)
    sys.path.insert(0, parentdir)
else:
    sys.path.insert(0, currentdir)
    print(currentdir)

# %% Logger
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# %% Import AI Libs
from dataparsing.parsing_functions import DataParsingStrategy
from dataparsing.output_strategies import DataOutputStrategy, convinAIentOutputStrategy
from dataparsing.parsing_strategy_selector import search_suitable_dataparsing_strategy_for_file
from dataparsing import parsing_functions

from anAIlysis import ESA
from anAIlysis import signal_processing as DSP
# from anAIlysis import plotting as eplt
from anAIlysis import signal_analysis as SA
from anAIlysis import get_samplerate as GSR
from anAIlysis import file_handler as fh

np.random.seed(seed=42)

# %% Init plotting
# Use WebAgg if no display is connected (e.g. ssh) and not running as notebook
import matplotlib
if (os.name == 'posix' and "DISPLAY" not in os.environ) and not isnotebook():
    matplotlib.use('WebAgg')
elif isnotebook():
    try:
        get_ipython().run_line_magic('matplotlib', 'widget')
    except ModuleNotFoundError:
        get_ipython().run_line_magic('matplotlib', 'inline')

# Return used Matplotlib backend
logger.info(f"Matplotlib backend: {matplotlib.get_backend()}")

import matplotlib.pyplot as plt

# %% [markdown]
# ## 3-Phase Current Sine Wave

# %% Create Current Data
model = pd.DataFrame(columns=['amplitude', 'frequency', 'theta', 'offset', 'fs', 'length', 'time', 'signal', 'noise', 'signal_noise'])

for i in range(0, 3):
    model.at[i, 'amplitude'] = 1 + np.random.random_sample()
    model.at[i, 'frequency'] = 50
    model.at[i, 'theta'] = i * 2 * np.pi / 3.0 + np.random.random_sample()
    model.at[i, 'offset'] = 0 + np.random.random_sample()
    model.at[i, 'fs'] = 10e3
    model.at[i, 'length'] = 10 # in seconds
    model.at[i, 'time'] = np.arange(0, model.at[i, 'length'], 1 / model.at[i, 'fs'])
    model.at[i, 'signal'] = model.at[i, 'amplitude'] * np.sin(2 * np.pi * model.at[i, 'frequency'] * model.at[i, 'time'] + model.at[i, 'theta']) + model.at[i, 'offset']
    
    model.at[i, 'mu'] = 0
    model.at[i, 'sigma'] = 0.1
    model.at[i, 'noise'] = np.random.normal(model.at[i, 'mu'], model.at[i, 'sigma'], len(model.at[i, 'signal']))
    model.at[i, 'signal_noise'] = model.at[i, 'signal'] + model.at[i, 'noise']

# %% Sine Wave

length_plot = int(0.1 * model.at[0, 'fs'])

fig, ax = plt.subplots()

for i in range(0, 3):
    ax.plot(model.at[i, 'time'][0:length_plot], model.at[i, 'signal_noise'][0:length_plot])

ax.plot(model.at[0, 'time'][0:length_plot], model.at[0, 'signal'][0:length_plot])
ax.set_title('Sine wave')
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude = sin(time)')

# %% [markdown]
# ## DQ-Transformation

# %% DQ Transform Ideal

i_a = model.at[0, 'amplitude'] * np.sin(2 * np.pi * model.at[0, 'frequency'] * model.at[0, 'time'] + 0 * model.at[1, 'theta'])
i_b = model.at[0, 'amplitude'] * np.sin(2 * np.pi * model.at[0, 'frequency'] * model.at[0, 'time'] + 1 * model.at[1, 'theta'])
i_c = model.at[0, 'amplitude'] * np.sin(2 * np.pi * model.at[0, 'frequency'] * model.at[0, 'time'] + 2 * model.at[1, 'theta'])

i_d, i_q = ESA.dq(i_a, i_b, i_c)

lim = 3

fig, ax = plt.subplots(figsize=(7,7))
hb = ax.hexbin(i_d, i_q, bins='log')
# plt.legend(loc="upper left")
plt.xlabel(r'$i_D$')
plt.ylabel(r'$i_Q$')
ax.set(xlim=(-lim,lim), ylim=(-lim,lim))
ax.set_aspect('equal', 'box')
ax.set_title(f"DQ-Transformation")
cb = fig.colorbar(hb, ax=ax, label='log10(N)')

# %% DQ Transform Noisy

i_d, i_q = ESA.dq(model.at[0, 'signal_noise'], model.at[1, 'signal_noise'], model.at[2, 'signal_noise'])

lim = 3

fig, ax = plt.subplots(figsize=(7,7))
hb = ax.hexbin(i_d, i_q, bins='log')
# plt.legend(loc="upper left")
plt.xlabel(r'$i_D$')
plt.ylabel(r'$i_Q$')
ax.set(xlim=(-lim,lim), ylim=(-lim,lim))
ax.set_aspect('equal', 'box')
ax.set_title(f"DQ-Transformation")
cb = fig.colorbar(hb, ax=ax, label='log10(N)')

# %% [markdown]
# ## Motor Current Signature Analysis (MCSA)

# %% Motor Current Signature Analysis (MCSA)
signal = model.at[0, 'signal_noise']
fs = model.at[0, 'fs']

freq, fft = ESA.csa(signal, fs=fs)

fig, ax = plt.subplots()
ax.plot(freq, fft)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'Amplitude')
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
ax.set_title(f"Motor Current Signature Analysis")

# %% [markdown]
# ## Enhanced Park's Vector Approach (EPVA)

# %% Enhanced Park's Vector Approach
signal = model.at[0, 'signal_noise']
fs = model.at[0, 'fs']

freq, fft = ESA.epva(i_d, i_q, fs=fs)

fig, ax = plt.subplots()
ax.plot(freq, fft)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'Amplitude')
ax.set_xlim((0, 3e2))
# ax.set_ylim(ylim)
ax.set_title(f"Enhanced Park's Vector Approach")

# %% [markdown]
# ## PicoScope Data Imports

import pandas as pd

data_path = 'data/SINUS.csv'

data_path = os.path.join(parentdir, data_path) if isnotebook() else data_path

data_df = pd.read_csv(data_path, index_col=0, header=[0, 1], sep=';', decimal=",")

# %% Plot Data
fig, ax = plt.subplots(figsize=(12, 4))
data_df.plot(ax=ax)
ax.set_xlabel('Time %s' % data_df.columns.names[1])
ax.set_ylabel('Amplitude %s' % data_df.columns[0][1])

# %% Use Signal from PicoScope
channels = data_df.columns.get_level_values(0).unique()
amplitude = data_df[channels[0]].values

# %% [markdown]
## Estimate 3-Phase Sine Model Parameters


# %% Estimate and create sine waves from model
signals = [model.at[i, 'signal_noise'] for i in range(0, 3)]
fs = model.at[0, 'fs']

est_model = SA.calc_model(signals, fs)

est_signals = {}
for i in range(0, 3):
    est_signals[i] = SA.get_signal_from_model(est_model[i], fs, end_time=0.1)

# %% Plot Data
fig, axs = plt.subplots(nrows=2, sharex=True)

for i in range(0, 3):
    axs[0].plot(est_signals[i][0:1000])

# ax.plot(model.at[0, 'time'][0:length_plot], model.at[0, 'signal'][0:length_plot])
axs[0].set_title('Sine wave with estimated parameters')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude = sin(time)')

for i in range(0, 3):
    axs[1].plot(model.at[i, 'signal_noise'][0:1000])

# ax.plot(model.at[0, 'time'][0:length_plot], model.at[0, 'signal'][0:length_plot])
axs[1].set_title('Sine wave with original parameters')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Amplitude = sin(time)')

plt.tight_layout()