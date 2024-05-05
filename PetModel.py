import torch.nn as nn

class PetModel(nn.Module):

    def __init__(self, classifier):
        super(PetModel, self).__init__()
        self.classifier = classifier

        # freeze classifier weights
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.fc_layers = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   #ouput of 2 for x and y coordinates
            nn.ReLU()
        )

    def forward(self, X):
        return self.fc_layers(self.classifier(X))