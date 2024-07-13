
import os
import logging
from functools import partial
from omegaconf import OmegaConf

import torch
from vocos import Vocos
from .model.dvae import DVAE
from .model.gpt import GPT_warpper
from .utils.gpu_utils import select_device
from .utils.infer_utils import count_invalid_characters, detect_language, apply_character_map, apply_half2full_map
from .utils.io_utils import get_latest_modified_file
from .infer.api import refine_text, infer_code

from huggingface_hub import snapshot_download

logging.basicConfig(level = logging.INFO)


class Chat(AudioProcessor):
    def __init__(self):
        super().__init__()

