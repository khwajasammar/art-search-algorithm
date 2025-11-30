
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import wget
import sys
import __main__
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models 
from torchvision.datasets import CIFAR10
from torchvision.models import squeezenet1_1
from torch.optim.lr_scheduler import StepLR

#new
#ref = sys.modules[__name__]
#batch_size refers to the amount of sample images to work through for every iteration
#num_epochs refers to # of epochs (# of times we go through training data in our algo) during training
#num_workers refers to how many workers (sub-processes) to use when loading data, theoretically should make CPU data loading more efficient
#learning_rate refers to a hyper-param that decides pace a/w algo updates, regulates weights based on loss gradient
#loss function tells us how far off our predictions are (predicted vs actual), attempt to minimize it by adjusting weights accordinly
#in TVGG arch, con. layers are connected
#com split up memory by task, diff workers take up diff tasks, more efficient
NUM_EPOCHS = 1
LEARNING_RATE = .001
PATH = 'models/model1.pth'
data_path = Path('/Users/sammarkhwaja/Desktop/algo1')
# Ensure the directory exists
os.makedirs(os.path.dirname(PATH), exist_ok=True)
FLAGS = {'datadir': '/Users/sammarkhwaja/Desktop/algo1', 'IMAGE_SIZE': 128, 'batch_size': 32, 'num_workers': 4}

def download_data(tbd):   
    if tbd:
        wget.download(url="https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
                    out = FLAGS['datadir'])
        # Uncompress the tar file
        import tarfile
        file = tarfile.open(data_path / "artbench-10-imagefolder-split.tar")
        file.extractall(data_path)
        file.close()
download_data(False)
# data_path = '/Users/sammarkhwaja/Downloads/artbench-10-batches3-py'
# wget.download(url="https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
#               out = FLAGS['datadir'])
# import tarfile
# file = tarfile.open(data_path / "artbench-10-imagefolder-split.tar")
# file.extractall(data_path)
# file.close() 

data_transform = transforms.Compose ([
transforms.Resize(size=(FLAGS['IMAGE_SIZE'], FLAGS['IMAGE_SIZE'])),
#transforms.TrivialAugmentWide(num_magnitude_bins=8), # Data Augmentation
transforms.ToTensor()
])
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#data_transform = transforms.Compose([
     #transforms.ToTensor(),  # Convert images to PyTorch tensors
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the pixel values
#])
#transforms.Resize(size=(FLAGS['IMAGE_SIZE'], FLAGS['IMAGE_SIZE'])),
#transforms.TrivialAugmentWide(num_magnitude_bins=8), # Data Augmentation
#transforms.ToTensor()
#])


#image_path = ('C:/Users/Sammar Khwaja/Downloads/artbench-10-batches-py')
image_path = Path('/Users/sammarkhwaja/Desktop/algo1/artbench-10-imagefolder-split')
#train_data = datasets.ImageFolder(root=image_path, transform=data_transform)
#test_data = datasets.ImageFolder(root=image_path, transform=data_transform)

#train data coming from "train" folder in "algo1" folder, test coming from "test" folder
train_data = datasets.ImageFolder(root=image_path / "train", transform=data_transform)
test_data = datasets.ImageFolder(root=image_path / "test", transform=data_transform)

train_DataLoader = DataLoader(dataset=train_data,
                              batch_size=FLAGS['batch_size'],
                              num_workers=FLAGS['num_workers'],
                              shuffle=True)
test_DataLoader = DataLoader(dataset=train_data,
                             batch_size=FLAGS['batch_size'],
                             num_workers=FLAGS['num_workers'],
                             shuffle=False)

#ensure images are sorted in right classes visually
class_name = 'impressionism'  # Replace ‘your_class_name’ with the actual class name you’re interested in
class_index = train_data.class_to_idx[class_name]
# Function to find and display images of a certain class
def find_and_display_images(dataset, class_index, num_images=5):
    found_images = 0
    for images, labels in DataLoader(dataset, batch_size=1, shuffle=True):  # Shuffle to get random images
        if labels.item() == class_index:
            plt.figure(figsize=(3, 3))
            plt.imshow(images[0].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            plt.title(f'Class: {class_name}')
            plt.axis('off')
            plt.show()
            found_images += 1
            if found_images == num_images:
                break
# Now, call the function to display images of the specified class
#find_and_display_images(train_data, class_index)

#output number of images per class in train and test
from collections import Counter
# For the training dataset
train_targets = [train_data.classes[label] for label in train_data.targets]
train_class_counts = Counter(train_targets)
# For the test dataset
test_targets = [test_data.classes[label] for label in test_data.targets]
test_class_counts = Counter(test_targets)
# Print the counts for each class in the training dataset
# print("Training Set Samples Per Class:")
# for class_name, count in train_class_counts.items():
#     print(f"{class_name}: {count}")
# # Print the counts for each class in the testing dataset
# print("\nTesting Set Samples Per Class:")
# for class_name, count in test_class_counts.items():
#     print(f"{class_name}: {count}")

#add way to classify warm/cool tone of image to user

#prevent variation w/ initial weights
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#build nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2) # Assuming 3-channel input images, 16 out after conv., kernel size 5 (breadth and height of 'filter'), stride = conv pace, paddings = adding 'padding' layers outside of input matrix data
        #nn.Conv2d = computes 2D convolution, filter/kernal slides over 2D input data, performs el-wise multiplication, sums result to out (5x5 matrices)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #nn.MaxPool2d = applies 2D max pooling over input signal (downsizes input along spatial dims and takes max value over in window of size kernal_size), similar to conv2d but does not have trainable weight like conv layer, suppresses noise in input data, takes max value in a section 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        #another conv layer, now with input channel (ex: RGB has 3 channels) of 16 and out of 32
        final_size = FLAGS['IMAGE_SIZE']//4 #NEW!
        self.fc1 = nn.Linear(32 * final_size * final_size, 120) # Adjust the size based on your image dimensions, 32 (8x8) feature maps
        #usually w/ fully connected layers. each in neuron connected to each out neuron
        #nn.Linear = applies lin. transform to in data using weigts and biases learned during training, in_features and out_features, size of weight mat is (out x in), size of bias vec is out, tensors match # features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # Assuming 10 classes
        #final linear transformation resulting in final 10 out_features (classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #relu = activation function applied after e/ layer, interperets pos part of argument, solves vanishing grad issues
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout(x)  # Apply dropout #NEW!!
        x = x.view(-1, 32 * FLAGS['IMAGE_SIZE']//4 * FLAGS['IMAGE_SIZE']//4) # Flatten the output for the fully connected layer
        #changing tensor shape w/o copying it
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        #returns final lin output layer to determine class (?)
#print("File name set to: {}" .format(__name__))

if __name__ == '__main__':
    model = SimpleCNN().to(device)
    #model training:
    optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
    #NEW!
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Adjust step_size and gamma as needed
    print('Start training')
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode
        running_loss = 0.0
        total_batches = len(train_DataLoader)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Total batches: {total_batches}')
        for batch_idx, (images, labels) in enumerate(train_DataLoader): ##???
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()  # Update the learning rate #NEW!
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx+1}/{total_batches}, Current Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Avg. Loss: {running_loss/total_batches:.4f}')
    #new, save model
    torch.save(model, PATH)
    model.eval()  # Set model to evaluation mode
    #model evaluation:
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_DataLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the test images: {100 * correct / total} %')

 

























