# RNN Code

Please note that this folder is specifically for the RNN only models listed in our Final Report. You can select which model to use in the yaml file in the exps/ directory


## To use the data

First, create a data folder `data/` and download `project_data.zip` and extract it to `data/`.  Next, run `python process_data.py` from this directory.

To load the data, use the `load_data()` function in `utils.py`.  It provides a dict with all the data in it.

## To use the training script

To train a model, run `python train_model.py`.  By default, this will use the config `exps/template.yaml`.  To use a different config, use `python train_model.py --config exps/[other_config.yaml]`.  

All options in the config can be overwritten via command line arguments.  To use a different learning rate for example, use
`python train_model.py --learning_rate 0.001`.
