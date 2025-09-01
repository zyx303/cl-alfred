import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional
from PIL import Image
import numpy as np

try:
    import clip
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False


class _CLIPViTL14_336(object):
    """
    CLIP ViT-L/14@336px image encoder wrapper
    """
    
    def __init__(self, args, eval=True, share_memory=False):
        if not _HAS_CLIP:
            raise ImportError("openai-clip not installed. Please install 'clip' from https://github.com/openai/CLIP")
        
        self.device = torch.device('cuda' if args.gpu else 'cpu')
        
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        
        if eval:
            self.model = self.model.eval()
            
        if share_memory:
            self.model.share_memory()
        
        # Get the visual encoder
        self.visual_encoder = self.model.visual
        
        # Output dimension for ViT-L/14@336px
        self.output_channels = 768  # CLIP ViT-L/14 has 768 dimensional features
        
    def extract(self, x):
        """
        Extract image features using CLIP visual encoder
        Args:
            x: preprocessed image tensor [B, 3, 336, 336]
        Returns:
            features: [B, 768] image features
        """
        x = x.to(self.visual_encoder.conv1.weight.dtype)
        with torch.no_grad():
            # Extract image features  (frame, hidden=768)
            image_features = self.visual_encoder(x)
            # Normalize features (CLIP uses L2 normalization)
            image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def encode_image(self, images: Union[torch.Tensor, List[Image.Image]]):
        """
        Encode images to features
        Args:
            images: Either preprocessed tensor or list of PIL Images
        Returns:
            features: normalized image features
        """
        
        if isinstance(images, list):
            # Preprocess PIL images
            processed = torch.stack([self.preprocess(img) for img in images])
            if self.device.type == 'cuda':
                processed = processed.to(self.device)
        else:
            processed = images
            
        return self.extract(processed)
    
    def featurize(self, images, batch=32):
        """
        Featurize images in batches (compatible with existing interface)
        """
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
            # Process PIL images
            images_normalized = torch.stack([self.preprocess(i) for i in images], dim=0)
        else:
            # Assume already preprocessed
            images_normalized = images
            
        if self.device.type == 'cuda':
            images_normalized = images_normalized.to(self.device)

        out = []
        with torch.set_grad_enabled(False):
            for i in range(0, images_normalized.size(0), batch):
                b = images_normalized[i:i+batch]
                out.append(self.extract(b))
        return torch.cat(out, dim=0)


def load_clip_model(model_name="ViT-L/14@336px", device="cuda"):
    """
    Load CLIP model with specified configuration
    """
    if not _HAS_CLIP:
        raise ImportError("openai-clip not installed. Please install 'clip' from https://github.com/openai/CLIP")
    
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess
