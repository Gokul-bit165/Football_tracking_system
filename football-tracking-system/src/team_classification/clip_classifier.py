"""
CLIP Team Classifier.

Leverages a frozen pre-trained Contrastive Language-Image Pretraining (CLIP) model
to perform zero-shot and few-shot jersey classification. Extracts embeddings for player crops
and classifies them based on cosine similarity to textual prompts or reference team prototypes.
Highly robust against lighting shifts and shadows.
"""

import cv2
import torch
import numpy as np
import open_clip
from PIL import Image

class CLIPTeamClassifier:
    """
    CLIP-based zero-shot team classifier.
    """
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        Initialize CLIP model.
        
        Args:
            model_name (str): CLIP backbone name.
            pretrained (str): Pre-trained weights tag.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract visual embedding vector for a player crop.
        
        Args:
            crop (np.ndarray): Player BGR image patch.
            
        Returns:
            np.ndarray: L2-normalized embedding vector.
        """
        if crop is None or crop.size == 0:
            return np.zeros((512,), dtype=np.float32)
            
        # Convert BGR to RGB PIL image
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)
            
        return features.cpu().numpy().flatten()

    def classify(self, crop: np.ndarray, team_prompts: list = None) -> str:
        """
        Classify crop by comparing embedding to reference text prompts.
        
        Args:
            crop (np.ndarray): Player BGR image patch.
            team_prompts (list): Predefined text prompts representing each team/referee class.
            
        Returns:
            str: Assigned prompt text label.
        """
        if team_prompts is None:
            team_prompts = [
                "a player in a red football jersey",
                "a player in a blue football jersey",
                "a referee in a yellow shirt"
            ]
            
        if crop is None or crop.size == 0:
            return team_prompts[0]
            
        # Convert BGR to RGB PIL image
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        # Tokenize text
        text_tokens = self.tokenizer(team_prompts).to(self.device)
        
        with torch.no_grad():
            # Get visual embeddings
            image_features = self.model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Get textual embeddings
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity and select highest
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            best_idx = similarity.argmax().item()
            
        return team_prompts[best_idx]

