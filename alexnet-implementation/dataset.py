# Import the libraries
from torchvision import datasets
from torchvision import transforms


# Data transformations
transformations = transforms.Compose([
                                    transforms.Resize((227, 227)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# Define the root directory
data_dir = './data'

# Train dataset
train_dataset = datasets.CIFAR10(root = data_dir, train = True, download = True, transform = transformations)

# Test dataset
test_dataset = datasets.CIFAR10(root = data_dir, train = False, download = True, transform = transformations)


