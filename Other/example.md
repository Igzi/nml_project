```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from seiz_eeg.dataset import EEGDataset

import os

```


```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

    cuda


# The data

We model *segments* of brain activity, which correspond to windows of a longer *session* of EEG recording.

These segments, and their labels, are described in the `segments.parquet` files, which can be directly loaded with `pandas`.


```python
# You might need to change this according to where you store the data folder
# Inside your data folder, you should have the following structure:
# data
# ├── train
# │   ├── signals/
# │   ├── segments.parquet
# │-- test
#     ├── signals/
#     ├── segments.parquet

from pathlib import Path


data_path = "/home/stnikoli/nml_project/data/"

DATA_ROOT = Path(data_path)
```


```python
print(data_path)
```

    /home/stnikoli/nml_project/data/


We have train segments, with labels...


```python
clips_tr = pd.read_parquet(DATA_ROOT / "train/segments.parquet")
display(clips_tr.iloc[100:115])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>start_time</th>
      <th>end_time</th>
      <th>date</th>
      <th>sampling_rate</th>
      <th>signals_path</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pqejgcpt_s002_t001_0</th>
      <td>0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_1</th>
      <td>0</td>
      <td>12.0</td>
      <td>24.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_2</th>
      <td>0</td>
      <td>24.0</td>
      <td>36.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_3</th>
      <td>0</td>
      <td>36.0</td>
      <td>48.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_4</th>
      <td>0</td>
      <td>48.0</td>
      <td>60.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_5</th>
      <td>0</td>
      <td>60.0</td>
      <td>72.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_6</th>
      <td>0</td>
      <td>72.0</td>
      <td>84.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_7</th>
      <td>0</td>
      <td>84.0</td>
      <td>96.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_8</th>
      <td>0</td>
      <td>96.0</td>
      <td>108.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_9</th>
      <td>0</td>
      <td>108.0</td>
      <td>120.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_10</th>
      <td>1</td>
      <td>120.0</td>
      <td>132.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_11</th>
      <td>1</td>
      <td>132.0</td>
      <td>144.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_12</th>
      <td>1</td>
      <td>144.0</td>
      <td>156.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_13</th>
      <td>1</td>
      <td>156.0</td>
      <td>168.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
    <tr>
      <th>pqejgcpt_s002_t001_14</th>
      <td>1</td>
      <td>168.0</td>
      <td>180.0</td>
      <td>2003-01-01</td>
      <td>250</td>
      <td>signals/pqejgcpt_s002_t001.parquet</td>
    </tr>
  </tbody>
</table>
</div>


... and test segments, unlabeled.


```python
clips_te = pd.read_parquet(DATA_ROOT / "test/segments.parquet")
display(clips_te.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start_time</th>
      <th>end_time</th>
      <th>date</th>
      <th>sampling_rate</th>
      <th>signals_path</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pqejgcvm_s001_t000_0</th>
      <td>0.0</td>
      <td>12.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_1</th>
      <td>12.0</td>
      <td>24.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_2</th>
      <td>24.0</td>
      <td>36.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_3</th>
      <td>36.0</td>
      <td>48.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_4</th>
      <td>48.0</td>
      <td>60.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_5</th>
      <td>60.0</td>
      <td>72.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_6</th>
      <td>72.0</td>
      <td>84.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_7</th>
      <td>84.0</td>
      <td>96.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_8</th>
      <td>96.0</td>
      <td>108.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
    <tr>
      <th>pqejgcvm_s001_t000_9</th>
      <td>108.0</td>
      <td>120.0</td>
      <td>2002-01-01</td>
      <td>250</td>
      <td>signals/pqejgcvm_s001_t000.parquet</td>
    </tr>
  </tbody>
</table>
</div>


The EEG signals are stored in separate files at the corresponding `signals_path`, and each of them contains all data from a given session. Next section illustrate a convenient interface for loading and iterating on data.


```python
clips_tr.shape
```




    (12993, 6)




```python
display(
    pd.read_parquet(DATA_ROOT / "train" / clips_tr.iloc[1992]["signals_path"]).iloc[
        12000:12015
    ]
)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FP1</th>
      <th>FP2</th>
      <th>F3</th>
      <th>F4</th>
      <th>C3</th>
      <th>C4</th>
      <th>P3</th>
      <th>P4</th>
      <th>O1</th>
      <th>O2</th>
      <th>F7</th>
      <th>F8</th>
      <th>T3</th>
      <th>T4</th>
      <th>T5</th>
      <th>T6</th>
      <th>FZ</th>
      <th>CZ</th>
      <th>PZ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12000</th>
      <td>146.516575</td>
      <td>57.710376</td>
      <td>-120.512372</td>
      <td>24.751375</td>
      <td>18.037504</td>
      <td>21.089264</td>
      <td>20.784088</td>
      <td>8.271874</td>
      <td>-39.945924</td>
      <td>-37.199341</td>
      <td>79.377868</td>
      <td>-133.024586</td>
      <td>-37.199341</td>
      <td>24.751375</td>
      <td>51.912033</td>
      <td>46.113691</td>
      <td>-54.289193</td>
      <td>46.113691</td>
      <td>-123.258956</td>
    </tr>
    <tr>
      <th>12001</th>
      <td>146.420203</td>
      <td>45.406967</td>
      <td>-114.810401</td>
      <td>21.298068</td>
      <td>20.687716</td>
      <td>20.077364</td>
      <td>24.044652</td>
      <td>9.396207</td>
      <td>-35.464657</td>
      <td>-36.075008</td>
      <td>82.028080</td>
      <td>-143.496939</td>
      <td>-36.380184</td>
      <td>25.570531</td>
      <td>53.036366</td>
      <td>49.069079</td>
      <td>-56.826972</td>
      <td>47.543199</td>
      <td>-121.524271</td>
    </tr>
    <tr>
      <th>12002</th>
      <td>121.717014</td>
      <td>11.548500</td>
      <td>-90.990616</td>
      <td>22.229658</td>
      <td>27.417649</td>
      <td>18.262371</td>
      <td>30.164232</td>
      <td>3.003574</td>
      <td>-18.358742</td>
      <td>-44.909049</td>
      <td>70.752632</td>
      <td>-136.156655</td>
      <td>-36.058947</td>
      <td>24.060714</td>
      <td>52.136900</td>
      <td>42.371270</td>
      <td>-31.176131</td>
      <td>54.273132</td>
      <td>-120.287506</td>
    </tr>
    <tr>
      <th>12003</th>
      <td>126.439210</td>
      <td>33.055373</td>
      <td>-87.183947</td>
      <td>23.289743</td>
      <td>26.646678</td>
      <td>17.491400</td>
      <td>28.782910</td>
      <td>2.842955</td>
      <td>-19.740065</td>
      <td>-46.290371</td>
      <td>62.352263</td>
      <td>-131.739634</td>
      <td>-42.628260</td>
      <td>22.679391</td>
      <td>45.567586</td>
      <td>37.938188</td>
      <td>-30.116047</td>
      <td>52.586633</td>
      <td>-121.974004</td>
    </tr>
    <tr>
      <th>12004</th>
      <td>111.855013</td>
      <td>40.138667</td>
      <td>-112.449303</td>
      <td>27.626453</td>
      <td>20.302231</td>
      <td>25.795398</td>
      <td>23.048814</td>
      <td>14.503888</td>
      <td>-32.493207</td>
      <td>-31.577679</td>
      <td>74.013196</td>
      <td>-133.506442</td>
      <td>-42.258837</td>
      <td>23.964342</td>
      <td>51.125001</td>
      <td>52.345705</td>
      <td>-46.531300</td>
      <td>50.819825</td>
      <td>-116.721766</td>
    </tr>
    <tr>
      <th>12005</th>
      <td>123.515946</td>
      <td>19.145775</td>
      <td>-102.619425</td>
      <td>29.826933</td>
      <td>23.723414</td>
      <td>28.606229</td>
      <td>22.807886</td>
      <td>16.094015</td>
      <td>-37.922126</td>
      <td>-33.649663</td>
      <td>65.532517</td>
      <td>-138.019834</td>
      <td>-41.889413</td>
      <td>30.437284</td>
      <td>49.358193</td>
      <td>56.377239</td>
      <td>-50.739515</td>
      <td>55.766887</td>
      <td>-116.352342</td>
    </tr>
    <tr>
      <th>12006</th>
      <td>116.143538</td>
      <td>16.961357</td>
      <td>-106.329722</td>
      <td>28.558043</td>
      <td>23.370052</td>
      <td>24.590756</td>
      <td>27.642515</td>
      <td>9.637135</td>
      <td>-20.880459</td>
      <td>-39.496191</td>
      <td>67.620563</td>
      <td>-137.457668</td>
      <td>-43.158303</td>
      <td>24.590756</td>
      <td>44.732368</td>
      <td>50.225535</td>
      <td>-28.204682</td>
      <td>58.160109</td>
      <td>-116.705704</td>
    </tr>
    <tr>
      <th>12007</th>
      <td>107.759230</td>
      <td>32.685949</td>
      <td>-102.812168</td>
      <td>28.718662</td>
      <td>20.173736</td>
      <td>22.920319</td>
      <td>27.497958</td>
      <td>6.745994</td>
      <td>-14.921497</td>
      <td>-41.166628</td>
      <td>64.729423</td>
      <td>-133.024586</td>
      <td>-47.575323</td>
      <td>18.037504</td>
      <td>43.672283</td>
      <td>45.198163</td>
      <td>-21.330192</td>
      <td>60.456960</td>
      <td>-117.765789</td>
    </tr>
    <tr>
      <th>12008</th>
      <td>110.505814</td>
      <td>39.094644</td>
      <td>-108.610511</td>
      <td>30.854894</td>
      <td>21.699615</td>
      <td>28.718662</td>
      <td>23.530671</td>
      <td>13.459865</td>
      <td>-30.790646</td>
      <td>-35.978637</td>
      <td>64.119071</td>
      <td>-132.109058</td>
      <td>-43.608036</td>
      <td>25.056551</td>
      <td>47.944746</td>
      <td>51.606858</td>
      <td>-43.913211</td>
      <td>55.268969</td>
      <td>-116.850261</td>
    </tr>
    <tr>
      <th>12009</th>
      <td>109.734843</td>
      <td>34.051210</td>
      <td>-129.523093</td>
      <td>31.304627</td>
      <td>16.351006</td>
      <td>32.220154</td>
      <td>20.623469</td>
      <td>18.182061</td>
      <td>-39.191015</td>
      <td>-31.256441</td>
      <td>77.386193</td>
      <td>-133.490381</td>
      <td>-37.665136</td>
      <td>28.558043</td>
      <td>49.615183</td>
      <td>58.770461</td>
      <td>-51.398053</td>
      <td>58.465285</td>
      <td>-112.738417</td>
    </tr>
    <tr>
      <th>12010</th>
      <td>105.799680</td>
      <td>18.824537</td>
      <td>-91.038801</td>
      <td>27.674639</td>
      <td>28.895343</td>
      <td>27.064287</td>
      <td>25.233232</td>
      <td>10.889962</td>
      <td>-26.036326</td>
      <td>-40.989947</td>
      <td>61.549168</td>
      <td>-135.289312</td>
      <td>-45.872762</td>
      <td>21.265944</td>
      <td>44.154140</td>
      <td>51.173186</td>
      <td>-32.445021</td>
      <td>63.685400</td>
      <td>-114.537349</td>
    </tr>
    <tr>
      <th>12011</th>
      <td>113.653945</td>
      <td>13.251061</td>
      <td>-87.151823</td>
      <td>25.763274</td>
      <td>26.678802</td>
      <td>23.932218</td>
      <td>27.289154</td>
      <td>7.147542</td>
      <td>-14.214774</td>
      <td>-42.290960</td>
      <td>55.975692</td>
      <td>-136.895502</td>
      <td>-48.699655</td>
      <td>18.439051</td>
      <td>40.716895</td>
      <td>48.041118</td>
      <td>-21.233821</td>
      <td>63.910266</td>
      <td>-114.312482</td>
    </tr>
    <tr>
      <th>12012</th>
      <td>104.321986</td>
      <td>12.158852</td>
      <td>-119.066802</td>
      <td>29.859056</td>
      <td>18.262371</td>
      <td>32.300464</td>
      <td>24.060714</td>
      <td>18.872723</td>
      <td>-22.020853</td>
      <td>-28.429548</td>
      <td>64.649113</td>
      <td>-134.020423</td>
      <td>-50.707392</td>
      <td>31.079760</td>
      <td>43.591974</td>
      <td>61.292178</td>
      <td>-34.838243</td>
      <td>60.681826</td>
      <td>-112.047755</td>
    </tr>
    <tr>
      <th>12013</th>
      <td>102.089383</td>
      <td>26.710926</td>
      <td>-119.468349</td>
      <td>35.866204</td>
      <td>19.691879</td>
      <td>35.561028</td>
      <td>18.776351</td>
      <td>22.438462</td>
      <td>-37.070846</td>
      <td>-25.168984</td>
      <td>56.618168</td>
      <td>-131.980563</td>
      <td>-52.634819</td>
      <td>31.288565</td>
      <td>44.105954</td>
      <td>67.299326</td>
      <td>-48.057179</td>
      <td>62.416511</td>
      <td>-108.482015</td>
    </tr>
    <tr>
      <th>12014</th>
      <td>121.154848</td>
      <td>23.803723</td>
      <td>-106.506403</td>
      <td>24.414075</td>
      <td>25.024427</td>
      <td>28.381362</td>
      <td>25.024427</td>
      <td>13.732917</td>
      <td>-26.245131</td>
      <td>-37.231465</td>
      <td>61.645540</td>
      <td>-139.465404</td>
      <td>-47.607447</td>
      <td>19.836436</td>
      <td>43.334983</td>
      <td>54.931669</td>
      <td>-35.095233</td>
      <td>63.476595</td>
      <td>-112.609922</td>
    </tr>
  </tbody>
</table>
</div>


## Loading the signals

For convenience, the `EEGDataset class` provides functionality for loading each segment and its label as `numpy` arrays.

You can provide an optional `signal_transform` function to preprocess the signals. In the example below, we have two bandpass filtering functions, which extract frequencies between 0.5Hz and 30Hz which are used in seizure analysis literature:

The `EEGDataset` class also allows to load all data in memory, instead of reading it from disk at every iteration. If your compute allows it, you can use `prefetch=True`.


```python
bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)


def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]
```


```python
# You can change the signal_transform, or remove it completely
dataset_tr = EEGDataset(
    clips_te,
    signals_root=DATA_ROOT / "test",
    signal_transform=fft_filtering,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)
```


```python
fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
for i, (x, y) in enumerate(dataset_tr):
    if i > 4:
        break
    axes[i].plot(x)
    axes[i].grid()
    axes[i].set(title=(f"Clip {i}: {y}"))

fig.tight_layout()
plt.show()
```


    
![png](output_15_0.png)
    



```python
fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
for i, ax in zip(range(108, 113), axes):
    x, y = dataset_tr[i]
    ax.plot(x)
    ax.grid()
    ax.set(title=(f"Clip {i}: {y}"))

fig.tight_layout()
plt.show()
```


    
![png](output_16_0.png)
    


## Compatibility with PyTorch

The `EEGDataset` class is compatible with [pytorch datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), which allow you to load batched data.


```python
# Dependences
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
```


```python
def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Torch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1)
```


```python
loader_tr = DataLoader(dataset_tr, batch_size=512, shuffle=True)
```


```python
for x, y in loader_tr:
    print(x.shape)
    print(y.shape)
    break
```

    torch.Size([512, 354, 19])
    torch.Size([512])


## Baseline: LSTM model for sequential data

In this section, we provide a simple baseline for the project using an LSTM model without any special optimization.


```python
# Define the model
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)  # Output for binary classification

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_dim]
        """
        out, (h_n, c_n) = self.lstm(x)  # out shape: [batch_size, seq_len, hidden_dim]
        last_timestep = out[:, -1, :]  # [batch_size, hidden_dim]
        logits = self.fc(last_timestep)  # [batch_size, 1]
        return logits
```


```python
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

    Using device: cuda



```python
# Instantiate model, loss, and optimizer, then move model to device
model = SimpleLSTM(input_dim=19, hidden_dim=64, num_layers=3, dropout=0.3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```


```python
# Training loop
num_epochs = 1000
train_losses = []

for epoch in tqdm(range(num_epochs), desc="Training"):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in loader_tr:
        # Move data to GPU (if available)
        x_batch = x_batch.float().to(device)  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.float().unsqueeze(1).to(device)  # [batch_size, 1]

        # Forward pass
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader_tr)
    train_losses.append(avg_loss)
```

    Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:30<00:00,  4.74it/s]



```python
# Plot the training loss
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(range(1, num_epochs + 1), train_losses, marker="o")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```


    
![png](output_27_0.png)
    


# Submission


```python
# Create test dataset
dataset_te = EEGDataset(
    clips_te,  # Your test clips variable
    signals_root=DATA_ROOT
    / "test",  # Update this path if your test signals are stored elsewhere
    signal_transform=fft_filtering,  # You can change or remove the signal_transform as needed
    prefetch=True,  # Set to False if prefetching causes memory issues on your compute environment
    return_id=True,  # Return the id of each sample instead of the label
)

# Create DataLoader for the test dataset
loader_te = DataLoader(dataset_te, batch_size=512, shuffle=False)
```


```python
# Generate the submission file for Kaggle

# Set the model to evaluation mode
model.eval()

# Lists to store sample IDs and predictions
all_predictions = []
all_ids = []

# Disable gradient computation for inference
with torch.no_grad():
    for batch in loader_te:
        # Assume each batch returns a tuple (x_batch, sample_id)
        # If your dataset does not provide IDs, you can generate them based on the batch index.
        x_batch, x_ids = batch

        # Move the input data to the device (GPU or CPU)
        x_batch = x_batch.float().to(device)

        # Perform the forward pass to get the model's output logits
        logits = model(x_batch)

        # Convert logits to predictions.
        # For binary classification, threshold logits at 0 (adjust this if you use softmax or multi-class).
        predictions = (logits > 0).int().cpu().numpy()

        # Append predictions and corresponding IDs to the lists
        all_predictions.extend(predictions.flatten().tolist())
        all_ids.extend(list(x_ids))

def remove_underlines(s):
    s = s.replace("___", "###")
    s = s.replace("_", "")
    s = s.replace("###", "_")
    return s

corrected_ids = [remove_underlines(i) for i in all_ids]
# Create a DataFrame for Kaggle submission with the required format: "id,label"
submission_df = pd.DataFrame({"id": corrected_ids, "label": all_predictions})

# Save the DataFrame to a CSV file without an index
submission_df.to_csv("submission.csv", index=False)
print("Kaggle submission file generated: submission.csv")
```

    Kaggle submission file generated: submission.csv

