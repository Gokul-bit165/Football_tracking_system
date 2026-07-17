"""
CNN Team Classifier.

A supervised lightweight CNN (e.g. ResNet18 backbone) designed for classifying player crops 
into jersey categories. Learns pattern and texture variations from training data,
providing low-latency supervised classification for a fixed set of team kits.
"""

import torch.nn as nn

class CNNTeamClassifier(nn.Module):
    """
    Lightweight supervised convolutional network for jersey classification.
    """
    def __init__(self, num_classes: int = 3):
        """
        Args:
            num_classes (int): Typically 3 (Team A, Team B, Referees).
        """
        super().__init__()
        pass

    def forward(self, x):
        """
        Forward pass for classification.
        """
        pass
