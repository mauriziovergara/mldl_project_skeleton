from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        # max pooling dopo ogni convolution nel forward layer
        # qui definisco solo la mia funzione come "pool" e la chiamo con kernel_size=2 e stride=2 perché voglio dimezzare la dimensione ad ogni passaggio
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Add more layers...
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # il fully-connected ha 256 elementi ma con dimensione h*w che partiva da 224 e con 3 maxpool quindi 224/2^3 = 28. Quindi dimensione finale considera anche 28*28.
        self.fc1 = nn.Linear(256 * 28 * 28, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.pool(x) # B x 64 x 112 x 112
        x = self.conv2(x).relu()
        x = self.pool(x)
        x = self.conv3(x).relu()
        x = self.pool(x)
        # alla fine prima del fully-connected layer fai un flatten
        x = x.flatten(start_dim=1) # B x 256
        x = self.fc1(x) # B x 200



        return x
