# Guidance-Based RNN Scaling
## Prerequisites: Setup + Installation

To run guidance, I usually use conda with the following installed from Anaconda. You also need to install torchvision and match the torch CUDA version.
* python=3.9
* pytorch=2.1.1

Afterwards, you can run `pip install -r requirements.txt` to install the remaining packages I use. 

You should be good to go! This code downloads the OpenWebText dataset for pretraining. If you need to change the location of that download, run
```
export HF_HOME=[PATH/TO/HOME]
```

You should also set this in under `language_modeling/webtext.py` as a `cache_dir` argument to `load_dataset`.

## Commands to run Guidance

The main current commands to run are under `pretrain.sh`. You can just run this shell script and it should start automatically. These commands will all use DDP to run everything across 2 GPUs. But, the code isn't very fast even so I think you'll want to use as many as you can. You can modify the number of GPUs to all the GPUs -- I have no idea how many there will be -- but make sure to divide the global batch size -- 256 -- by the number of GPUs. So with 4 GPUs, `--batch_size` should be set to 64. 

The first time the code is run, it will reload the dataset for OpenWebText, chunk the data, and save this as .arrow files to disk. Afterwards, it won't do this again unless you specify. 

If any argument in `pretrain.sh` doesn't make sense, let me know.

The current networks are set to RNNs the size of GPT-2 Medium. In theory we would like to run GPT-2 Large as well. To run an RNN (guided or unguided) at the size of GPT-2 Large, set `--hidden_dim 6800`, `embedding_dim 6800`, and `num_layers 4`. In this case, I actually have no idea how large the guide should be? Maybe it should be slightly scaled up to the size of GPT-2 Small (so double the number of layers in the command?). To run the Transformer at GPT-2 Large, run `d_model 1280` and `trans_layers 36`. 