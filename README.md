# Character-Level-RNN

Character Level RNN feeds the text character-by-character (individual letter) into the RNN. This project aims to accomplish following two things:

- [Classification](#classification)
- [Generation](#generation)


## Classification

For classification, I've written [model_c.py](./model_c.py) where there are two models: `RNN` model which is a somewhat scratch model and `T_RNN` model which uses PyTorch's built in RNN.

### Training

Here we have two models, which can be trained as

**a. Scratch RNN**
```python
python train_c.py
```

**b. PyTorch RNN**

```python
python train_c.py --model_type pytorch
```

### Evaluate

For evaluating both type of network, we can use following scripts.

**a. Scratch RNN**

```python
python evaluate_c.py --name Najjar
```
```Output: The name Najjar is: Arabic```

**b. PyTorch RNN**

```python
python evaluate_c.py --name Palmeiro --model_type pytorch
```
```Output: The name Palmeiro is: Portuguese```

### Results

The training results of the run for 100_000 iterations is shown below.

**a. Scratch Classification RNN**
<img src="results/scratch-classification.png" alt="Scratch Classification RNN" style="width:100%;">

**b. PyTorch RNN**
<img src="results/pytorch-classification.png" alt="PyTorch Classification RNN" style="width:100%;">


## Generation

