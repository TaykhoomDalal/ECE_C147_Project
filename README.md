# ECE_C147_Project

Papers to read:

[Deep learning with convolutional neural networks for brain mapping and decoding of movement-related information from the human EEG](https://arxiv.org/pdf/1703.05051.pdf)

[EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces](https://arxiv.org/pdf/1611.08024.pdf)

## To use the data

First, create a data folder `data/` and download `project_data.zip` and extract it to `data/`.  Next, run `python process_data.py` from this directory.

To load the data, use the `load_data()` function in `utils.py`.  It provides a dict with all the data in it.


## To use the training script

To train a model, run `python train_model.py`.  By default, this will use the config `exps/template.yaml`.  To use a different config, use `python train_model.py --config exps/[other_config.yaml]`.  

All options in the config can be overwritten via command line arguments.  To use a different learning rate for example, use
`python train_model.py --learning_rate 0.001`.