# [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

Example of using Knowledge Distillation with the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset

## Teacher Network:
* The teacher Neural Net architecture has 2 convolutional layers and 2 fully connected layers.
* Teacher will stop training when number of errors on test set is 70 or less

## Student Network
* The student Neural Net architecture has 2 fully connected layers with 300 hidden units.
* The training will try different temperatures.

## Install dependencies:
```
pip install -r requirements.txt
```

## Run training for both teacher and student
```
python train.py
```

## Results
![results](https://github.com/dariocazzani/knowledge-distill/blob/master/images/tau.png)
