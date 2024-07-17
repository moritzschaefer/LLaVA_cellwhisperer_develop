"""
This wraps the image tower, which we don't use because we create a dedicated dataset with transcriptome embeddings.

I anyways introduced the (untested) necessary changes, in case we want to use this class later
"""
from pathlib import Path
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

# from cellwhisperer.jointemb.cellwhisperer_lightning import TranscriptomeTextDualEncoderLightning
# from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
# from cellwhisperer.config import get_path, model_path_from_name


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            raise NotImplementedError('Delay load not implemented')
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):

        model_path = Path(self.vision_tower_name).expanduser()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(model_path)
        pl_model.eval().to(device)
        pl_model.model.prepare_models(
            pl_model.model.transcriptome_model, pl_model.model.text_model, modes_to_freeze=["L", "U", "u"]
        )
        pl_model.freeze()

        # TODO transcriptome_processor_kwargs might be missing
        processor = TranscriptomeTextDualEncoderProcessor(
            pl_model.model.transcriptome_model.config.model_type,
            model_path_from_name(pl_model.model.text_model.config.model_type),
        )

        tokenizer = processor.tokenizer
        self.image_processor = processor.transcriptome_processor

        self.vision_tower = pl_model.model  # TODO this includes both models
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_layer == -1:  # get shared embeddings
            image_features = image_forward_outs[1]
        elif self.select_layer == -2:  # get image block output (features)
            image_features = image_forward_outs[0]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            raise NotImplementedError('Patch not implemented')
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower.get_transcriptome_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower.get_transcriptome_features(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
