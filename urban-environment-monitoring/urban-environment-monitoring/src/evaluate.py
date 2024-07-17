import torch
from torch.utils.data import DataLoader
from src.data_loader import SatelliteDataset
from src.model import UNet

def evaluate():
    dataset = SatelliteDataset(root_dir='data/processed', transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = UNet(in_channels=3, out_channels=4)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    with torch.no_grad():
        for images in dataloader:
            outputs = model(images)
            # Implement evaluation logic here

if __name__ == "__main__":
    evaluate()
