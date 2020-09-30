"""Problem 3 - Training on MNIST"""
import numpy as np
from mytorch.nn.sequential import Sequential
from mytorch.nn.linear import Linear
from mytorch.optim.sgd import SGD
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.nn.activations import *
from mytorch.tensor import Tensor
# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    model = Sequential(Linear(784,20),ReLU(),Linear(20,10))
    optimizer = SGD(model.parameters(),lr=0.1)
    criterion = CrossEntropyLoss()
    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(model,optimizer,criterion,train_x,train_y,val_x,val_y)
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    model.train()
    store_validation_accuracy = []
    store_loss = []
    Avg_loss = []
    for j in range(num_epochs):
        p = np.random.permutation(len(train_x))
        shuffled_x = train_x[p] 
        shuffled_y = train_y[p]
        xx = np.split(shuffled_x,len(shuffled_x)/BATCH_SIZE)
        yy = np.split(shuffled_y,len(shuffled_y)/BATCH_SIZE)
        for i, (batch_data,batch_labels) in enumerate(zip(xx,yy)):
            optimizer.zero_grad() # clear any previous gradients
            out = model(Tensor(batch_data))
            loss = criterion(out,Tensor(batch_labels))
            store_loss.append(loss.data)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                Avg_loss.append(np.mean(np.array(store_loss)))
                store_loss = []
                accuracy = validate(model,val_x,val_y)
                store_validation_accuracy.append(accuracy)
                model.train()
    print(Avg_loss)
    return store_validation_accuracy
                

def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    model.eval()
    xx = np.split(val_x,len(val_x)/BATCH_SIZE)
    yy = np.split(val_y,len(val_y)/BATCH_SIZE)
    num_correct = 0
    for i, (batch_data,batch_labels) in enumerate(zip(xx,yy)):
        out = model(Tensor(batch_data))
        batch_preds = np.argmax(out.data,axis=1)
        num_correct_i = np.sum(batch_labels==batch_preds)
        num_correct += num_correct_i
    accuracy = num_correct / len(val_y)
    return accuracy 

