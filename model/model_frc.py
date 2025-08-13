import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_norm(input, axis=-1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    
    return output

class FaceRaceClassifier(nn.Module):
    def __init__(self):
        super(FaceRaceClassifier, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 6)
        self.fc3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = l2_norm(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)

        return x
