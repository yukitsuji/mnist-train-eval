# MNIST Train Eval

Train script and evaluation script for MNIST dataset.
The code is based on `pytorch/examples` (https://github.com/pytorch/examples/tree/master/mnist).


## Installation
```
pip install -r requirements.txt
```


## Train
Train a CNN with the training dataset.

### Command
```
python train.py --out model.pkl
```
### Inputs
- `--batch-size`
- `--epochs`
- `--lr`

### Outputs
- `--out`: A pickle file containing the trained model.

## Eval
Calculate the accuracy of the predictions with the validation dataset.

### Command
```
python eval.py --model model.pkl --out acc.txt
```
### Inputs
- `--model`: A pickle file containing a trained model.

### Outputs
- `--out`: A text file containing the accuracy (e.g. 0.9876).
