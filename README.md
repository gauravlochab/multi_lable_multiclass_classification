# multi_lable_multiclass_classification

## Usage:

1. **Preprocessing and training on GPU**<br>
  [training notebook](https://github.com/gauravlochab/multi_lable_multiclass_classification/blob/main/multiclass_classification.ipynb).<br>

2. **Inference File** <br>
-> Run the script as illustrated in the notebook above (in last cell).<br>
```bash 
$ python inference.py --csv_file path_to_test_csv --ckpts path_to_model_weights
```
colab example given that you have added the inference file,test.csv from repo to colab and downloaded weights from link below
```gdown --id 1-D4QzRDhtlFqj4wzZUj4s9TpS2xA369t```
```bash 
$ python inference.py --csv_file /content/attributes_test.csv --ckpts /content/drive/MyDrive/outputs/model.pth
```
-> This will generate [`Output.csv`]([https://github.com/gauravlochab/multi_lable_multiclass_classification/blob/main/Output.csv]) file in the same directory as of inference.py

## Models
Under the `outputs` directory, model weights and loss graph are available.
Drive link to model  : [model.pt](https://drive.google.com/file/d/1-D4QzRDhtlFqj4wzZUj4s9TpS2xA369t/view?usp=sharing)


