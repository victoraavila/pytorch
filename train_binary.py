import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from calculate_mean_and_std import calculate_mean_and_std
import os

def mnist_binary_loss(predictions: torch.Tensor, labels: torch.Tensor):
    # Getting the predictions of being class 'label_1' for all 64 images
    predictions_class_label_1 = predictions[:, -1]

    # Taking all predictions of being class 'label_1' for 64 images to range [0, 1] by calling sigmoid()
    # Important: sigmoid(x) only depends on one single number x, not on other numbers to be calculated
    predictions_class_label_1_sigmoid = predictions_class_label_1.sigmoid()

    # If probabilities can only be in range [0, 1], if the probability(class 'label_1') = x, probability(class 'label_0') = 1 - x
    # If the image represents an 'label_1', fill the labels Tensor with the percentage of confidence left in its prediction
    # If the image does not represent an 'label_1', fill the labels Tensor with the percentage of confidence it had in predicting 'label_0' (which is the percentage of confidence left in predicting 'label_0')
    # Important: 1 - sigmoid(class 'label_1') != sigmoid(class 'label_0'), but I think it assumes so for simplification, since the two classes are mutually exclusive and exhaustive
    return torch.where(labels == label_1, 1 - predictions_class_label_1_sigmoid, predictions_class_label_1_sigmoid).mean()

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten_layer = nn.Flatten()
        self.fc1_layer = nn.Linear(in_features = 784, out_features = 128) # out_features is the hidden_size. The smaller the size, the greater the compression.
        self.relu_layer = nn.ReLU()
        self.fc2_layer = nn.Linear(in_features = 128, out_features = 2) # Classifies into 2 classes

    # Defining the forward pass of the neural network
    def forward(self, x):
        x = self.flatten_layer(x)
        x = self.fc1_layer(x)
        x = self.relu_layer(x)
        x = self.fc2_layer(x)
        return x
    
# Defining the device for inference
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


train_dataset = datasets.MNIST(root='.', download=False, train=True) # PIL Image
mean, std = calculate_mean_and_std(train_dataset.data) # Finding dynamically that the mean value for pixel in [0.0, 1.0] is 0.1307 and std for pixel in [0.0, 1.0] is 0.3081

# We need to apply all the transformations at the same time in order to not overwrite the 1st transformation with the 2nd transformation
train_dataset.transform = transforms.Compose([
    transforms.ToTensor(), # Convert PIL Image HxWxC [0, 255] to Tensor CxHxW [0.0, 1.0]
    transforms.Normalize((mean), (std)) # This is necessary in order to converge faster with gradient descent. image = (image - mean) / std for each channel.
])

# Leaving only label_0s and label_1s in both train dataset and val dataset
label_0, label_1 = 0, 8
# train_dataset.targets = 60000 integers from 0 to 9 indicating the label
# (train_dataset.targets==label_0) | (train_dataset.targets==label_1) = 60000 bools indicating whether the label is in (0, label_1) or not in (0, label_1)
train_dataset_indexes_to_keep = (train_dataset.targets==label_0) | (train_dataset.targets==label_1)
# Leaving only the labels 0 and label_1 in the targets Tensor
train_dataset.targets = train_dataset.targets[train_dataset_indexes_to_keep]
# Leaving only the data points related to the labels 0 and label_1 in the data Tensor
train_dataset.data = train_dataset.data[train_dataset_indexes_to_keep]

# Leaving only 0s and 8s
val_dataset = datasets.MNIST(root='.', train=False)
val_dataset.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean), (std))
])
# val_dataset.targets = 10000 integers from 0 to 9 indicating the label
# (val_dataset.targets==label_0) | (val_dataset.targets==label_1) = 10000 bools indicating whether the label is in (0, 8) or not in (0, label_1)
val_dataset_indexes_to_keep = (val_dataset.targets==label_0) | (val_dataset.targets==label_1)
# Leaving only the labels label_0 and label_1 in the targets Tensor
val_dataset.targets = val_dataset.targets[val_dataset_indexes_to_keep]
# Leaving only the data points related to the labels label_0 and label_1 in the data Tensor
val_dataset.data = val_dataset.data[val_dataset_indexes_to_keep]

# Defining the loader, which is a iterable that passes batches of data points during training
# batch_size should fit in the GPU and at the same time be small sufficient so the parameters are effectively learned
train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False)

# Bug fix
del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']

# Instantiating model and optimizer
model = SimpleNet().to(device = device)
optimizer = optim.SGD(params = model.parameters(), lr = 1e-4)

# Training loop
number_of_epochs = 30
for epoch in range(number_of_epochs):
    model.train() # Sets the model to training mode (the opposite is model.eval())
    epoch_loss, number_of_images_seen, number_of_correct_predictions = 0, 0, 0

    for images, labels in train_loader:
        # Loading the batch into the GPU
        images, labels = images.to(device = device), labels.to(device = device) # Changes images.device and labels.device, not the images.dtype
        
        # Forward pass
        outputs = model(images) # outputs has shape [64, 2] (it is a binary classification) with results in range [-1, 1]
        loss = mnist_binary_loss(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad() # Throw out the previous gradients in order to not use them in the current iteration (step) when calling loss.backward()
        loss.backward() # x.grad += dloss/dx for every parameter x which has requires_grad = True
        optimizer.step() # Updates the parameters. x += -lr * x.grad for Stochastic Gradient Descent (SGD)

        # Variables necessary for training metrics
        epoch_loss += loss.item() # Get the loss of the current iteration (step) and sum it to the loss of the current epoch
        number_of_images_seen += labels.size(0) # Accumulates the size of the 0th dimension (labels.shape = [64])
        _, predicted_class = torch.max(outputs.data, 1) # Returns [greatest logit among the 2 classes, index of the class with greatest logit among the 2 classes] (2 Tensors)
        predicted_class = torch.where(predicted_class == 1, torch.tensor(label_1), torch.tensor(label_0))
        prediction_was_correct = predicted_class == labels # Returns a [64] Tensor with False/True telling if the prediction was correct
        number_of_correct_predictions += prediction_was_correct.sum().item() # Gets an integer with the quantity of correct predictions in this iteration

    # Get this epoch's training metrics
    epoch_avg_loss = epoch_loss / len(train_loader) # The avg loss of this epoch is the accumulated loss for all steps of this epochs divided by the num of iterations (steps)
    epoch_accuracy = 100 * (number_of_correct_predictions / number_of_images_seen) # The accuracy of this epoch is the accumulated number of correct predicitions divided by the num of images
    print(f"Epoch {epoch + 1}")
    print(f"The average loss in this epoch was {epoch_avg_loss:.4f}")
    print(f"Train accuracy in this epoch was {epoch_accuracy:.4f}")

    # Get this epoch's validation metrics
    model.eval()
    number_of_images_seen, number_of_correct_predictions = 0, 0
    for images, labels in val_loader:
        images, labels = images.to(device = device), labels.to(device = device)
        outputs = model(images)

        number_of_images_seen += labels.size(0)
        _, predicted_class = torch.max(outputs.data, 1)
        predicted_class = torch.where(predicted_class == 1, torch.tensor(label_1), torch.tensor(label_0))
        prediction_was_correct = predicted_class == labels
        number_of_correct_predictions += prediction_was_correct.sum().item()

    epoch_accuracy = 100 * (number_of_correct_predictions / number_of_images_seen)
    print(f"Val accuracy in this epoch was {epoch_accuracy:.4f}")
    print("==================")