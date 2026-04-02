import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Model
# ------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ------------------------
# Load model ONCE (IMPORTANT)
# ------------------------
model = CNN()
model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
model.eval()

classes = ['paper', 'rock', 'scissors']

# ------------------------
# Predict function
# ------------------------
def predict_image(image):
    img = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

    # Normalize same as training
    img = (img / 255.0 - 0.5) / 0.5

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    return classes[pred.item()]