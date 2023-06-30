# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import random
import re

import pytest
import torch
from composer.loggers import InMemoryLogger
from omegaconf import OmegaConf as om

from llmfoundry.callbacks import ModelGauntlet


def mock_logger_keys(model_gauntlet_cfg, skip_keys):
    keys = []
    for cat in model_gauntlet_cfg['categories']:

        for b in cat['benchmarks']:
            if skip_keys and random.random() < 0.3:
                continue
            keys.append(
                f"metrics/{b['name']}/{b['num_fewshot']}-shot/InContextLearningAccuracy"
            )
    return keys


class MockState:

    def __init__(self):
        self.timestamp = 0


def get_callback(skip_keys=False):
    with open('scripts/eval/yamls/model_gauntlet.yaml', 'r') as icl_f:
        model_gauntlet_cfg = om.load(icl_f)
    model_gauntlet = model_gauntlet_cfg.model_gauntlet
    model_gauntlet.benchmark_sizes = None
    model_gauntlet.logger_keys = mock_logger_keys(
        model_gauntlet_cfg.model_gauntlet, skip_keys)
    model_gauntlet_callback = ModelGauntlet(**model_gauntlet)
    return model_gauntlet_callback


def mock_logger_metrics(model_gauntlet_callback, tgt_acc=0.5):
    logger = InMemoryLogger()
    pat = re.compile(r'metrics/(.*?)/(\d+)-shot(/.*?)?/InContextLearning(.*)')

    random_baselines = {
        b['name']: b['random_baseline']
        for cat in model_gauntlet_callback.categories for b in cat['benchmarks']
    }

    for key in model_gauntlet_callback.logger_keys:
        match = pat.match(key)
        eval_name = match.group(1)
        if model_gauntlet_callback.subtract_random_baseline and model_gauntlet_callback.rescale_accuracy:
            desired_acc = random_baselines[eval_name] + (
                1 - random_baselines[eval_name]) * tgt_acc
        else:
            desired_acc = tgt_acc

        logger.data[key] = [(0, torch.tensor(desired_acc))]

    return logger


def test_gauntlet_all_metrics_present():
    model_gauntlet_callback = get_callback()
    logger = mock_logger_metrics(model_gauntlet_callback)
    state = MockState()
    logger.state = state
    model_gauntlet_callback.eval_end(state, logger)

    for cat in model_gauntlet_callback.categories:
        metric = f"metrics/model_gauntlet/{cat['name']}"
        assert logger.data[metric][0][1] == pytest.approx(0.5)
    assert logger.data['metrics/model_gauntlet/average'][0][1] == pytest.approx(
        0.5)


def test_gauntlet_some_metrics_absent():
    # this test sees if the scores are still scaled properly if some of the tasks are missing in the logger
    model_gauntlet_callback = get_callback(skip_keys=True)
    logger = mock_logger_metrics(model_gauntlet_callback, tgt_acc=0.75)
    state = MockState()
    logger.state = state
    model_gauntlet_callback.eval_end(state, logger)

    for cat in model_gauntlet_callback.categories:
        metric = f"metrics/model_gauntlet/{cat['name']}"
        assert logger.data[metric][0][1] == pytest.approx(0.75)
    assert logger.data['metrics/model_gauntlet/average'][0][1] == pytest.approx(
        0.75)
