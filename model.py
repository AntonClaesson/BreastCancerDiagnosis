import torch.nn as nn
from unet import UNet
from unet.unet_parts import Down

class ClassificationHead(nn.Module):
    def __init__(self, n_classes):
        super(ClassificationHead, self).__init__()
        self.extra_down = nn.Sequential(
            nn.MaxPool2d(2), #2 stride, 2 kernel size
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * 7 * 7, out_features=2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=n_classes)
        )
        #Note:
        #Feature vector to FC layer will depend on image input size.
        # If we increase image size, we use more memory but we also need to increase the FC layer requiring even
        # more memory. Therefore we run out of memory easily if we do that.
        # We could add additional conv+pool layers to downsample further and make the FC
        # layer smaller but it is a hard trade-off because that can reduce performance.
        # We don't want additional conv layers for the classifcation compared to the segmentation
        # therefore we just test 224x224

    def forward(self, x):
        x = self.extra_down(x)
        x = self.fc(x)
        return x


class DiagnosisModel(nn.Module):
    def __init__(self):
        super(DiagnosisModel, self).__init__()
        self.unet = UNet(n_channels=1, n_classes=2)
        self.classification_head = ClassificationHead(n_classes=2)

    def forward(self, x):
        features, semantic_logits = self.unet(x)
        cls_logits = self.classification_head(features)
        return {'cls_logits': cls_logits, 'semantic_logits': semantic_logits}

