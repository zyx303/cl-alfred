import torch
import torch.nn as nn
from torchvision import models, transforms
from autoaugment import ImageNetPolicy

# Optional: OpenAI CLIP for ViT-L/14@336px
try:
    import clip  # from openai/CLIP
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False


class Resnet18(object):
    '''
    pretrained Resnet18 from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model = models.resnet18(pretrained=True)

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

    def extract(self, x):
        return self.model(x)


class Resnet50(object):
    '''
    pretrained Resnet50 from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model = models.resnet50(pretrained=True)

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.output_channels = 2048

    def extract(self, x):
        return self.model(x)


class MaskRCNN(object):
    '''
    pretrained MaskRCNN from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, min_size=224):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, min_size=min_size)
        self.model = self.model.backbone.body
        self.feat_layer = 3

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()


    def extract(self, x):
        features = self.model(x)
        return features[self.feat_layer]


class Resnet(object):

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True, autoaug=False):
        self.model_type = args.visual_model
        self.gpu = args.gpu

        # choose model type
        if self.model_type == "maskrcnn":
            self.resnet_model = MaskRCNN(args, eval, share_memory)
        elif self.model_type == 'resnet18':
            self.resnet_model = Resnet18(args, eval, share_memory, use_conv_feat=use_conv_feat)
        elif self.model_type == 'resnet50':
            self.resnet_model = Resnet50(args, eval, share_memory, use_conv_feat=use_conv_feat)
        elif self.model_type in {'clip_vitl14_336', 'clip_vit_l_14_336'}:
            assert _HAS_CLIP, "openai-clip not installed. Please install 'clip' from https://github.com/openai/CLIP"
            self.resnet_model = _CLIPViTL14_336(args, eval, share_memory)
        else:
            raise ValueError(f"Unknown visual_model: {self.model_type}")

        # normalization transform
        if isinstance(self.resnet_model, _CLIPViTL14_336):
            # use CLIP's own preprocess
            self.transform = self.resnet_model.preprocess
        else:
            self.transform = self.get_default_transform(autoaug)


    @staticmethod
    def get_default_transform(autoaug=False):
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]

        if autoaug:
            transform_list.insert(1, ImageNetPolicy())

        return transforms.Compose(transform_list)

    def featurize(self, images, batch=32):
        images_normalized = torch.stack([self.transform(i) for i in images], dim=0)
        if self.gpu:
            images_normalized = images_normalized.to(torch.device('cuda'))

        out = []
        with torch.set_grad_enabled(False):
            for i in range(0, images_normalized.size(0), batch):
                b = images_normalized[i:i+batch]
                out.append(self.resnet_model.extract(b))
        return torch.cat(out, dim=0)


class _CLIPViTL14_336(object):
    """
    Wrapper to expose CLIP ViT-L/14@336px patch-token features as [B, C, N].
    C is the transformer width (e.g., 1024), N is number of patches (24x24=576).
    """

    def __init__(self, args, eval=True, share_memory=False):
        # load model and preprocess
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device="cuda" if args.gpu else "cpu")
        if eval:
            self.model.eval()
        if share_memory:
            try:
                self.model.share_memory()
            except Exception:
                pass
        self.gpu = args.gpu

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return patch embeddings (without CLS) as [B, C, N].
        """
        visual = self.model.visual
        # follow CLIP forward but retain all tokens
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, N]
        x = x.permute(0, 2, 1)  # [B, N, width]
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, cls.shape[0], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # ln_post is applied to class token in CLIP; we apply to all tokens for stability
        x = visual.ln_post(x)
        if hasattr(visual, 'proj') and visual.proj is not None:
            x = x @ visual.proj
        # drop CLS, keep patches
        x = x[:, 1:, :]  # [B, N, C]
        x = x.permute(0, 2, 1).contiguous()  # [B, C, N]
        return x

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._encode_patches(x)
