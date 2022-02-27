# ECE_C147_Project

Papers to read:

[Deep learning with convolutional neural networks for brain mapping and decoding of movement-related information from the human EEG](https://arxiv.org/pdf/1703.05051.pdf)

[EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces](https://arxiv.org/pdf/1611.08024.pdf)

## To use the data

First, create a data folder `data/` and download `project_data.zip` and extract it to `data/`.  Next, run `python process_data.py` from this directory.

To load the data, use the `load_data()` function in `utils.py`.  It provides a dict with all the data in it.