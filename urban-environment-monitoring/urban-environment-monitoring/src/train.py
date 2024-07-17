import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_loader import SatelliteDataset
from src.model import UNet

def train():
    dataset = SatelliteDataset(root_dir='data/processed', transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = UNet(in_channels=3, out_channels=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        for images in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == "__main__":
    train()
