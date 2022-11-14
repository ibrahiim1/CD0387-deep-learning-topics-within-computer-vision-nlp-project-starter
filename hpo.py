#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time

import argparse

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    model.eval()
    running_loss= 0
    corrects= 0
    sampels= 0
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs= model(inputs)
        loss= criterion(outputs, labels)
        
        running_loss+= loss.item() * inputs.size(0)
        corrects+= torch.sum(outputs==labels).item()
        sampels+= len(inputs) 
    
    total_loss = running_loss/sampels
    accuracy = corrects/sampels
    
    print(f"Total loss: {total_loss}, Accuracy is: {accuracy}")
    

def train(model, train_loader, valid_loader, criterion, optimizer, epoches, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    

    
    data = {"train": train_loader, "valid": valid_loader}

    for e in range(epoches):
        start= time.time()

        for mode in ["train", "valid"]:
            print(f"starting epoche: {e+1}, phase ---> {mode}")
            
            running_loss= 0
            corrects= 0
            sampels= 0
            
            
            for inputs, labels in data[mode]:
                inputs, labels = inputs.to(device), labels.to(device)

                if mode= "train":
                    model.train()
                    optimizer.zero_grad()

                else:
                    model.eval

                outputs= model(inputs)
                loss= criterion(outputs, labels)

                if mode= "train":
                    loss.backward()
                    optimizer.step()

                running_loss+= loss.item() * inputs.size(0)
                corrects+= torch.sum(outputs==labels).item()
                sampels+= len(inputs)
            total_loss = running_loss/sampels
            accuracy = corrects/sampels

            print(f"epoch: {e+1}, phase --> {mode}, total loss --> {total_loss}, accuracy --> {accuracy}")

        e_time = time.time() - start
        print(f"Epoche --> {e+1}, time ---> {e_time} seconds")



    
def net(device):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model= models.resnet50(pretrained= True)

    for para in model.parameters():
        para.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 133)

    for para in model.fc.parameters():
        para.requires_grad = True

    model = model.to(device)
    print(model)
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    args=parser.parse_args()
    
    main(args)
