from torch import nn

class Model(nn.Module):
    def __init__(self, output_dim):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(512, output_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return self.policy(x), self.value(x)

