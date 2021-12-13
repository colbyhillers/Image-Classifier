from torchvision.datasets import MNIST
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import classifier_methods

# load the images in the dataset and preprocess each image using the transforms
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]) # the normalization subtracts each pixel by the mean of images (0.1307) and divides by the standard deviation (0.3081)

train_dataset = MNIST("./", download = True, train = True, transform = transform)
test_dataset = MNIST("./", download = True, train = False, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 128, shuffle= True)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle= True)

images_dict = {}

for batch in train_loader:
  x, y = batch

  for label in range(10):
    if label in images_dict.keys():
      continue

    img_label = x[y == label]
    if len(img_label) > 0:
      images_dict[label] = img_label[0][0]

f, axarr = plt.subplots(2,5)
axarr[0,0].imshow(images_dict[0], cmap='gray')
axarr[0,1].imshow(images_dict[1], cmap='gray')
axarr[0,2].imshow(images_dict[2], cmap='gray')
axarr[0,3].imshow(images_dict[3], cmap='gray')
axarr[0,4].imshow(images_dict[4], cmap='gray')
axarr[1,0].imshow(images_dict[5], cmap='gray')
axarr[1,1].imshow(images_dict[6], cmap='gray')
axarr[1,2].imshow(images_dict[7], cmap='gray')
axarr[1,3].imshow(images_dict[8], cmap='gray')
axarr[1,4].imshow(images_dict[9], cmap='gray')

# Training/Evaluation of Classifier
epochs = 20
lr = 0.1
weights, biases = train(train_loader, epochs, lr)

acc = predict(weights, biases, test_loader)
print("Accuracy on test dataset: {}", acc)