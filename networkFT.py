from torch import nn
"""
Calculated output feature map size is given in [B, C, H, W] form
"""
class Net(nn.Module):
    
    def __init__(self, n_classes=10):
        super().__init__()
        self.convolutions = nn.Sequential( 
            nn.Conv2d(3, 4, 3, padding=1), # [B, 4, 240, 240]
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # [B, 4, 120, 120]

            nn.Conv2d(4, 8, 5, padding=1), # [B, 8, 118, 118]
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2), # [B, 8, 59, 59]
            
            nn.Conv2d(8, 16, 5, padding=1), # [B, 16, 59, 59]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),# [B, 16, 28, 28]   

            nn.Conv2d(16, 32, 3, padding=1), # [B, 32, 28, 28] 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),# [B, 32, 14, 14]   
        )
      
        self.fully_connected = nn.Sequential(
            nn.Linear(32*14*14, 16*14*14), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.3),
            nn.Linear(16*14*14, 1000), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.5),
            nn.Linear(1000, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, n_classes), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, img):

        # Apply convolution operations
        x = self.convolutions(img)
        # Reshape
        x = x.view(x.size(0), -1)
        # Apply fully connected operations
        x = self.fully_connected(x)

        return x
