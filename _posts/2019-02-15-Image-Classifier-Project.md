---
layout: post
title:  "Udacity Data Science Nanodegree : Image Classifier Project"
tags: udacity python deep-learning pytorch 
---


### Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, we might want to include an image classifier in a smart phone app. To do this, we'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, we will train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice we'd train this classifier, then export it for use in our application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, we can see a few examples below. 

![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/image_classifier/Flowers.png)


The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on our dataset
* Use the trained classifier to predict image content


```python
# Imports packages
%matplotlib inline
%config InlineBackend.figure_format = 'retina' 

import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import time

import torch 
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

from collections import OrderedDict
```

## 1 - Load the data

Here we'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise we can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, we'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. We'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this we don't want any scaling or rotation transformations, but we'll need to resize then crop the images to the appropriate size.

The pre-trained networks we'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets we'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```


```python
# Define our transforms for the training, validation, and testing sets
```


```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
}
```


```python
# TODO: Load the datasets with ImageFolder
```


```python
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}
```


```python
# TODO: Using the image datasets and the trainforms, define the dataloaders
```


```python
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=50,
                                             shuffle=True)
              for x in ['train', 'valid','test']}

dataset_sizes = {x: len(image_datasets[x])
                 for x in ['train', 'valid', 'test']}

class_names = image_datasets['train'].classes
```

### 2 - Label mapping


```python
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

### 3 - Visualize images


```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
n_images = 5

# Make a grid from batch
out = torchvision.utils.make_grid(inputs[0:n_images])

imshow(out, title=[class_names[x] for x in classes[0:n_images]])
```

<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/image_classifier/output_50_1.png"  width="603" height="182">



### 4 - Building and training the classifier


```python
# Load a pre-trained network 
model = models.vgg16(pretrained=True)
```


```python
model
```




    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace)
        (2): Dropout(p=0.5)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace)
        (5): Dropout(p=0.5)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )




```python
# Freeze parameters
for param in model.parameters():
    param.requires_grad = False
```

> After several experiments, in particular with different activation functions (Stochastic gradient descent, Adam optimizer) , different inputs, hidden dimensions and learning rates, here are the most efficient parameters having the best ratio between 
calculation time and accuray (and also, preventing overfitting)


```python
# Create a customer classifier and replace the VGG one
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
```


```python
# Set criterion and optimizier 
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
```


```python
#Checking CUDA
if torch.cuda.is_available():
    print('Using: GPU')
    device = torch.device("cuda:0")
    model.cuda()
    
else:
    print('Using: CPU')
    device = torch.device("cpu")
```

    Using: GPU
    


```python
#Source : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
```


```python
#Train a model with a pre-trained network
```


```python
def train_model(model, criterion, optimizer, num_epochs=8):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            if phase == 'valid':
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```


```python
trained_model = train_model(model, criterion, optimizer, num_epochs=8)
```

    Epoch 1/8
    ----------
    train Loss: 3.2378 Acc: 0.3893
    valid Loss: 0.9770 Acc: 0.7445
    
    Epoch 2/8
    ----------
    train Loss: 1.6457 Acc: 0.5726
    valid Loss: 0.6540 Acc: 0.8313
    
    Epoch 3/8
    ----------
    train Loss: 1.4578 Acc: 0.6221
    valid Loss: 0.6015 Acc: 0.8411
    
    Epoch 4/8
    ----------
    train Loss: 1.3857 Acc: 0.6427
    valid Loss: 0.5297 Acc: 0.8606
    
    Epoch 5/8
    ----------
    train Loss: 1.2770 Acc: 0.6735
    valid Loss: 0.4606 Acc: 0.8839
    
    Epoch 6/8
    ----------
    train Loss: 1.3279 Acc: 0.6616
    valid Loss: 0.4750 Acc: 0.8826
    
    Epoch 7/8
    ----------
    train Loss: 1.2768 Acc: 0.6703
    valid Loss: 0.5033 Acc: 0.8778
    
    Epoch 8/8
    ----------
    train Loss: 1.2880 Acc: 0.6792
    valid Loss: 0.4636 Acc: 0.8839
    
    Training complete in 34m 23s
    

### 5 - Testing the neural network


```python
# Validation on the test set
```


```python
def validate_model(model, dataloader, criterion):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = round(correct/total * 100, 2)
            
    return test_loss, accuracy
```


```python
test_loss, accuracy = validate_model(trained_model, dataloaders['test'], criterion)
print("Accuracy on test set : {:.2f}".format(accuracy))
```

    Accuracy on test set : 87.30
    

## 6 - Save the checkpoint


```model.class_to_idx = image_datasets['train'].class_to_idx```


```python
# Save the checkpoint 
```


```python
trained_model.to('cpu')
trained_model.class_to_idx = image_datasets['train'].class_to_idx
```


```python
checkpoint = {'state_dict': trained_model.state_dict(),
              'class_to_idx': trained_model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
```

### 7 - Loading the checkpoint


```python
# Create a function that loads a checkpoint and rebuilds the model
```


```python
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
        
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.load_state_dict = checkpoint['state_dict']
    
    return model
```


```python
# Reload the model from checkpoint
model = load_checkpoint('checkpoint.pth')
```


### 8 - Image Preprocessing

```python
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    # Building image transform
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    np_image = np.array(pil_image)    
    
        
    # Normalize values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image
```

```python
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.transpose(image, (1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
```


```python
imshow(process_image("flowers/test/15/image_06360.jpg"));
```


![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/image_classifier/output_43_0.png)


### 9 - Class Prediction

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
# TODO: Implement the code to predict the class from an image file
```


```python
def predict(image_path, model, topk=5):
        
    # Set model to evaluate
    model.eval()

    # Extract and convert to numpy array to tensor for PyTorch
    np_array = process_image(image_path)
    tensor_image = torch.from_numpy(np_array)
    tensor_image = tensor_image.float() 
    
    tensor_image = tensor_image.unsqueeze(0)

    #Checking device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    tensor_image.to(device)

    with torch.no_grad():
        output = model.forward(tensor_image.cuda())  

    # Reverse log conversion to cancel out the LogSoftMax
    output = torch.exp(output)
    
    # Predict topk probabilities, categories and labels
    topk_probs, topk_indexes = torch.topk(output, topk) 
    
    # Converting ouputs to lists
    topk_probs = topk_probs.tolist()[0]
    topk_indexes = topk_indexes.tolist()[0]
    
    idx_to_cat = {val: key for key, val in model.class_to_idx.items()}
    
    top_cats = [idx_to_cat[index] for index in topk_indexes ]
    
    top_labels = [cat_to_name[cat] for cat in top_cats ]

    return topk_probs, top_labels, top_cats
```


```python
image_path = 'flowers/valid/100/image_07931.jpg'
probas, labels,classes = predict(image_path, model)

print(probas)
print(labels)
print(classes)
```

    [0.018851974979043007, 0.01851687952876091, 0.018456503748893738, 0.01599038764834404, 0.013933357782661915]
    ['fire lily', 'columbine', 'lotus lotus', 'cyclamen', 'siam tulip']
    ['21', '84', '78', '88', '39']
    

### 10 - Sanity Checking

Now that we can use a trained model for predictions, we check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs.

![png](https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/image_classifier/inference_example.png)


```python
# Display an image along with the top 5 classes
```


```python
probas, labels, classes = predict('flowers/test/95/image_07487.jpg', model, topk = 5)
print('Probabilites of top categories:')
print(probas)
print('\nLabels of top categories:')
print(labels)
print('\n-------------------------------------------------------------------------------------------------------------')

fig, (ax1, ax2) = plt.subplots(figsize=(10,6), ncols=2)
_= imshow(process_image('flowers/test/95/image_07487.jpg'), ax = ax1)

# Plot Probability Distribution
y_pos = np.arange(len(labels))
ax2.barh(y_pos, probas, align='center', ecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.invert_yaxis()  
ax2.set_xlabel('Probability')
ax2.set_title('Probabilities of top 5 Flowers')
fig.tight_layout()
plt.show()
```

    Probabilites of top catagories:
    [0.01498053502291441, 0.013880652375519276, 0.013731810264289379, 0.012534408830106258, 0.012495827861130238]
    
    Labels of top catgories:
    ['bird of paradise', 'mexican aster', 'cyclamen', 'fire lily', 'water lily']
    
    -------------------------------------------------------------------------------------------------------------
    


<img src="https://raw.githubusercontent.com/amdmh/amdmh.github.io/master/_posts/img/image_classifier/output_50_1.png"  width="711" height="423">


