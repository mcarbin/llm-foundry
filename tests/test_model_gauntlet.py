import os
import random
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

from llmfoundry.utils.callbacks import ModelGauntlet

def get_callback():
    with open('scripts/eval/yaml/model_gauntlet.yaml', 'r') as icl_f:
        model_gauntlet_cfg = om.load(icl_f)
    model_gauntlet = model_gauntlet_cfg.model_gauntlet
    model_gauntlet.benchmark_sizes = None
    model_gauntlet_callback = ModelGauntlet(**model_gauntlet)
    return model_gauntlet_callback

def test_gauntlet():
    model_gauntlet_callback = get_callback()
    breakpoint()
    model_gauntlet_callback.eval_end()
