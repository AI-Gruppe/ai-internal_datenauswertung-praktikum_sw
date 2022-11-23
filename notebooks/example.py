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
parentdir = os.path.dirname(os.path.dirname(currentdir))
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
# from anAIlysis import signal_analysis as SA
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
# ## Example Plot
# $$f(x) = \sin(x)$$
# %% Plot sine wave

fs = 1e3
length = 1 # in seconds
time = np.arange(0, length, 1 / fs)

theta = 0
frequency = 10
amplitude = 1
sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)

plt.figure()
plt.plot(time, sinewave)
plt.title('Sine wave')
plt.xlabel('Time')
plt.ylabel('Amplitude = sin(time)')
plt.show()

# %% [markdown]
# ## Example Analysis
# ### Spectrum
# $$\mathcal{F}\{f(x)\}$$
# %% Plot sine wave frequency spectrum

freq, fft = DSP.power_fft(sinewave, fs)
plt.figure()
plt.plot(freq, fft)
plt.title('Sine wave')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# %% Multiple frequencies
# Time array
fs = 1e3
length = 1 # in seconds
time = np.arange(0, length, 1 / fs)

# First sine wave
theta = 0
frequency = 10
amplitude = 1
sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)

# Second sine wave
amplitude = 0.5
frequency = 20
sinewave += amplitude * np.sin(2 * np.pi * 2 * frequency * time + theta)

# Noise
amplitude = 0.1 # TODO Change to 1
noise = amplitude * np.random.randn(len(sinewave))
sinewave_noise = sinewave + noise

# Frequency response
freq, fft = DSP.power_fft(sinewave_noise, fs)

# %% Plot the result als subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 4))
axs[0].plot(time, sinewave_noise)
axs[0].set_title('Time signal')
axs[1].plot(freq, fft)
axs[1].set_title('Frequency spectrum')
plt.tight_layout()

# %% [markdown]
# ## Spectrogram

# %% Spectrogram without noise
D = librosa.stft(sinewave, hop_length=256, n_fft=1024)  # High resolution STFT
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Spectrogram without noise')
fig.colorbar(img, ax=ax, format="%+2.f dB")

# %% Spectrogram with noise
D = librosa.stft(sinewave_noise, hop_length=256, n_fft=1024)  # High resolution STFT
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Spectrogram without noise')
fig.colorbar(img, ax=ax, format="%+2.f dB")

# %% [markdown]
# ### Compose multiple sine waves
fs = 1e3
length = 10 # in seconds
time = np.arange(0, length, 1 / fs)

# First sine wave
theta = 0
frequency = 10
amplitude = 1
sinewaves = amplitude * np.sin(2 * np.pi * frequency * time + theta)

frequency = 2*frequency
new_wave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
sinewaves = np.concatenate((sinewaves, new_wave), axis=0)

frequency = 2*frequency
new_wave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
sinewaves = np.concatenate((sinewaves, new_wave), axis=0)

# %% Spectrogram with noise
D = librosa.stft(sinewaves, hop_length=256, n_fft=1024)  # High resolution STFT
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Spectrogram without noise')
fig.colorbar(img, ax=ax, format="%+2.f dB")

# %% [markdown]
# ## Example Analysis
# ### RMS
# $$x_{\mathrm{RMS}} = \sqrt{\frac{1}{n} \sum_{i=1}^n{x_i^2}}$$

# %% RMS
rms_clean = DSP.rms(sinewave)
rms_noise = DSP.rms(noise)
rms_combined = DSP.rms(sinewave_noise)

logger.info("RMS clean: {:.2f}".format(rms_clean))
logger.info("RMS noise: {:.2f}".format(rms_noise))
logger.info("RMS combined: {:.2f}".format(rms_combined))

# %% [markdown]
# ## Trend Analysis

# %% Trend analysis
rms = []

theta = 0
frequency = 10
amplitude = 1

for i in range (0, 100):
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    rms.append(DSP.rms(sinewave))
    amplitude += 0.1 * np.random.randn()

plt.figure()
plt.plot(rms)
plt.title('RMS Trend Analysis')
plt.xlabel('Time')
plt.ylabel('Amplitude = RMS(time)')
plt.show()

# %% [markdown]
# ## Audio Data
audio_path = os.path.abspath('../data/2022-10-23T17_22_44.027Z_rain.wav')
# y, sr = librosa.load(librosa.ex('trumpet'))
y, fs = librosa.load(audio_path)

D = librosa.stft(y, hop_length=256, n_fft=1024)  # High resolution STFT
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Spectrogram without noise')
fig.colorbar(img, ax=ax, format="%+2.f dB")

# %% 
ipd.Audio(y, rate=fs)

# %% Pandas Dataframe
rms_df = pd.DataFrame(columns=['RMS', 'Frequency', 'Amplitude'])

rms_df['RMS'] = rms
rms_df['Frequency'] = frequency
rms_df['Amplitude'] = amplitude


# %% [markdown]
# ## Image Data