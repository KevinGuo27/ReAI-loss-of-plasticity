
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import argparse

from pipeline.cnn import SimpleCNN
from pipeline.label_transform import multi_hot, one_hot, continuous_embedding

from analysis.activations import register_layer_hooks
from analysis.dead_relu_tracker import track_dead_neurons
from analysis.effective_rank import compute_effective_rank

# ---------------------------
# Parse label mode argument
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--label-mode', type=str, default='raw',
                    choices=['raw', 'one_hot', 'multi_hot', 'embedding'])
args = parser.parse_args()
label_mode = args.label_mode
print(f"Using label mode: {label_mode}")

# ---------------------------
# Model Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# Hook into internal layers
# ---------------------------
activations = register_layer_hooks(model, ["relu1", "relu2", "relu3"])

# ---------------------------
# Dataset Setup
# ---------------------------
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# ---------------------------
# Label Transformation + Loss Setup
# ---------------------------
if label_mode == "raw":
    transform_labels = lambda x: x
    criterion = nn.CrossEntropyLoss()
elif label_mode == "one_hot":
    transform_labels = one_hot
    criterion = nn.BCEWithLogitsLoss()
elif label_mode == "multi_hot":
    transform_labels = multi_hot
    criterion = nn.BCEWithLogitsLoss()
elif label_mode == "embedding":
    transform_labels = continuous_embedding
    criterion = nn.MSELoss()
else:
    raise ValueError(f"Unsupported label mode: {label_mode}")

# ---------------------------
# Training Loop
# ---------------------------
print("Starting training loop...")
for epoch in range(5):
    print(f"Epoch {epoch+1} starting...")
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        labels = transform_labels(labels).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} total loss: {running_loss:.3f}")

    for name, act in activations.items():
        dead, total = track_dead_neurons(act)
        rank = compute_effective_rank(act)
        print(f"[{name}] Dead: {dead}/{total} | Effective Rank: {rank:.2f}")

# # transform = transforms.ToTensor()
# # trainset = torchvision.datasets.CIFAR10(
# #     root="./data", train=True, download=True, transform=transform
# # )
# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# # for images, labels in trainloader:
# #     print(images.shape, labels)
# #     break


# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch import nn, optim
# from pipeline.cnn import SimpleCNN
# from pipeline.label_transform import multi_hot, one_hot, continuous_embedding

# from analysis.activations import register_layer_hooks
# from analysis.dead_relu_tracker import track_dead_neurons
# from analysis.effective_rank import compute_effective_rank


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleCNN().to(device)

# #USE THIS FOR ONE-HOT (nvm...)
# #criterion = nn.CrossEntropyLoss()  # Can adapt if labels aren't one-hot

# #USE THIS FOR MULTI-HOT and ALL others 
# criterion = nn.BCEWithLogitsLoss()


# optimizer = optim.Adam(model.parameters(), lr=0.001)

# activations = register_layer_hooks(model, ["relu1", "relu2", "relu3"])

# print("Transforming Labels...")
# transform = transforms.ToTensor()
# trainset = torchvision.datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform
# )
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# print("Training loop about to start...")

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--label-mode', type=str, default='raw', choices=['raw', 'one_hot', 'multi_hot', 'embedding'])
# args = parser.parse_args()
# label_mode = args.label_mode


# for epoch in range(5):
#     running_loss = 0.0
#     for inputs, labels in trainloader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         #CHANGE LABEL REPRESENTATION HERE (uncomment to change between...)
        
#         labels = multi_hot(labels).to(device)
#         # labels = one_hot(labels).to(device)
#         # labels = continuous_embedding(labels).to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     print(f"Epoch {epoch+1} loss: {running_loss:.3f}")

#     for name, act in activations.items():
#         dead, total = track_dead_neurons(act)
#         rank = compute_effective_rank(act)
#         print(f"[{name}] Dead: {dead}/{total} | Effective Rank: {rank:.2f}")