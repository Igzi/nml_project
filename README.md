#  EE-452: Network Machine Learning
# Graph-based EEG Analysis for Seizure Detection


<p align="center">
  <a href="#about">About</a> •
  <a href="#data">Data</a> •
  <a href="#method">Method</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#code-structure">Code Structure</a> •
  <a href="#results">Results</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
</p>

## Team
The project is accomplished by team:

Igor Pavlovic - @Igzi

Strahinja Nikolic - @strajdzsha

Milica Vukasinovic - @milicavukaa

Marija Rakonjac - @marijarakonjac

## About

This repository contains the work done for our [Network Machine Learning (EE-452)](https://edu.epfl.ch/coursebook/en/network-machine-learning-EE-452) project at EPFL. The goal of the project is to explore the use of graph-based methods for EEG signal processing, with a focus on seizure detection. EEG recordings are represented as time series captured from multiple electrodes placed on the scalp, and we evaluate both graph-based models and non-graph-based baselines.
We use a subset of the Temple University Hospital EEG Seizure Corpus (TUSZ), and compare different architectures on their ability to detect seizures from 12-second EEG segments. Our work includes extensive experimentation, performance analysis, and participation in a Kaggle competition hosted for the course.


## Data

We use a subset of the [Temple University Hospital EEG Seizure Corpus (TUSZ)](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00083/full), one of the largest publicly available EEG datasets for seizure detection.

- **Patients**: The dataset includes recordings from 50 patients for training and 25 for testing.
- **Sampling**: EEG signals are recorded from 19 electrodes at 250 Hz using the standard 10–20 system.
- **Segments**: The recordings are divided into non-overlapping 12-second windows.
- **Labels**: Each window in the training set is labeled as either normal brain activity or seizure activity.
- **Graphs**: For graph-based approaches, electrodes are modeled as nodes, and edges are constructed based on 3D distances between electrodes (provided via `distances_3d.csv`). 

> ⚠️ Usage of the dataset is subject to the TUSZ data license agreement. Redistribution is not permitted, and any attempt to re-identify anonymized subjects is strictly prohibited.


## Method

The goal of this project is to compare graph-based and non-graph-based methods for seizure detection from EEG signals. We evaluate multiple deep learning architectures to understand the benefits of modeling spatial relationships between electrodes.

### Approaches:

1. **Graph Neural Networks (GNNs)**:
   - We use architectures such as Graph Attention Networks (GAT) and Spatio-Temporal Graph Convolutional Networks (ST-GCN).
   - EEG signals are modeled as graphs, where each node represents an electrode and edges represent physical proximity or learned functional connectivity.
   - Graph structures are based on 3D distances between electrodes (`distances_3d.csv`) and visualized using the 10–20 EEG layout.

2. **Non-graph-based Baselines**:
   - We implement standard models such as 1D CNNs and LSTMs, treating EEG signals as multivariate time series.
   - These models serve as a baseline for comparing performance and interpretability against GNNs.

3. **Feature Extraction**:
   - Hjorth parameters (Activity, Mobility, Complexity) are computed as classical EEG features.
   - Experiments are conducted with and without feature extraction to test its influence on model performance.

4. **Training & Evaluation**:
   - All models are trained on labeled 12-second EEG windows.
   - Evaluation is done via cross-validation on the training set and submission to a private Kaggle leaderboard.
   - Models are implemented using PyTorch and PyTorch Geometric.

### Objective:

By comparing these approaches, we aim to highlight the potential of graph-based methods in capturing spatial dependencies and improving classification accuracy for seizure detection.


## How To Use

Each model (e.g., CNN, GAT, STGCN) is implemented and trained through a dedicated Jupyter notebook located in its corresponding folder. These notebooks handle training, evaluation, and submission file generation.

### Running the Models

To run an experiment:

1. Navigate to the model directory (e.g., `CNN/`, `GAT/`, `STGCN/`).
2. Open the notebook (e.g., `cnn.ipynb`) in Jupyter or VS Code.
3. Run all cells to:
   - Train the model
   - Evaluate it using validation data
   - Generate a CSV file for Kaggle submission

Example (for CNN model):

```bash
cd CNN
jupyter notebook cnn.ipynb
```

## Code Structure

```
│   .gitignore                # Specifies files and directories to be ignored by Git
│   adjacency_correlation.csv # Correlation-based adjacency matrix
│   evaluation.py             # Script to evaluate trained models
│   hjorth_features.py        # Script for computing Hjorth parameters
│   initial_sem_embs.txt      # Initial semantic embeddings (not actively used in this project)
│   requirements.txt          # List of required Python packages for running the notebooks
│   README.md                 # Project documentation
│   start_jupyter_server.bash # Bash script to launch a local Jupyter server (optional)
│
├───CNN                       # Convolutional Neural Network (time-domain and frequency-domain approaches)
│       cnn.ipynb
│       cnn_submission_fft.csv
│       cnn_submission_time.csv
│
├───ChebNet                   # Spectral GNN using Chebyshev polynomial filters
│       chebnet.ipynb
│       submission_seed5.csv
│
├───DCRNN                     # Diffusion Convolutional Recurrent Neural Network for spatio-temporal modeling
│       dcrnn.ipynb
│       submission_dcrnn.csv
│
├───GAT                       # Graph Attention Network for seizure detection using EEG graphs
│       GAT.ipynb
│
├───Hjorth                    # Models using Hjorth parameters (Activity, Mobility, Complexity)
│       hjorth.ipynb
│       hjorth.py
│       submission_hjorth_rf.csv
│
├───Other                     # Miscellaneous models and experiments (e.g., MLP, Transformer)
│       MLP.ipynb
│       Transformer.ipynb
│       example.ipynb
│       example.md
│
├───RNN                       # Recurrent Neural Network models (LSTM, BiLSTM)
│       BiLSTM copy.ipynb
│       BiLSTM.ipynb
│       RNN-full-sequence.ipynb
│       RNN.ipynb
│       submission_seed1.csv
│
├───STGCN                     # Spatio-Temporal Graph Convolutional Network
│       STGCN.ipynb
│       submission_full_sequence_256.csv
│       submission_full_sequence_512.csv
│       submission_seed1.csv
│       submission_smote_full_sequence.csv
│
├───Spectrogram               # Model based on spectrogram features of EEG signal
│       spectrogram.ipynb
│
├───data                      # EEG graph metadata and resources
│       distances_3d.csv
│
```

## Results



## Credits

This project was carried out as part of the [EE-452: Network Machine Learning](https://edu.epfl.ch/coursebook/en/network-machine-learning-EE-452) course at EPFL. We would like to thank the course staff for their guidance and support throughout the project.

We also acknowledge the creators of the [Temple University Hospital EEG Seizure Corpus (TUSZ)](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00083/full) for providing access to the dataset used in this work. Use of the dataset complies with the terms outlined in the official TUSZ agreement.

All models were implemented using open-source libraries including PyTorch, PyTorch Geometric, and scikit-learn.


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
