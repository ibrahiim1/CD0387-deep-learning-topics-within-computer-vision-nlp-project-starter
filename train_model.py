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
from torchvision import datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device):
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
        _, preds = torch.max(outputs, 1)
        running_loss+= loss.item() * inputs.size(0)
        corrects+= torch.sum(preds==labels.data).item()
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

                if mode== "train":
                    model.train()

                else:
                    model.eval()

                outputs= model(inputs)
                loss= criterion(outputs, labels)
                _, preds= torch.max(outputs, 1)

                if mode== "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss+= loss.item() * inputs.size(0)
                corrects+= torch.sum(preds==labels.data).item()
                sampels+= len(inputs)
            total_loss = running_loss/sampels
            accuracy = corrects/sampels

            print(f"epoch: {e+1}, phase --> {mode}, total loss --> {total_loss}, accuracy --> {accuracy}")

        e_time = time.time() - start
        print(f"Epoche --> {e+1}, time ---> {e_time} seconds")
    return model
    
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

def create_data_loaders(args, mode):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    batch_size= args.batch_size
    data_path= os.path.join(args.data, mode)

    if mode == "train":
        transform = transforms.Compose([transforms.Resize(255), transforms.RandomResizedCrop(224), 
                                        transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(), 
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                          (0.229, 0.224, 0.225))])

    else:
        transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                          (0.229, 0.224, 0.225))])

                                                        
    data= datasets.ImageFolder(data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size= batch_size, shuffle= True)
    return data_loader

def main(args):

    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader= create_data_loaders(args, "train")
    valid_loader= create_data_loaders(args, "valid")
    test_loader= create_data_loaders(args, "test")

    '''
    TODO: Initialize a model by calling the net function
    '''
    
    print(f"model using ---> {device} <---")

    model=net(device)
    
    '''
    TODO: Create your loss and optimizer
    '''

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr= args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, valid_loader, criterion, optimizer, args.epoches, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model_2.pt"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''

    parser.add_argument("--batch_size", type= int, default= 128)
    parser.add_argument("--lr", type= float, default= .01)
    parser.add_argument("--epoches", type= int, default= 14)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    
    main(args)
