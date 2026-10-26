import random

import pytest
import transformers

import lm_eval.api.metrics as metrics
from lm_eval.models.hf_vlms import HFMultimodalLM


def test_multimodal_auto_model_class_compatibility():
    expected = getattr(
        transformers,
        "AutoModelForVision2Seq",
        getattr(transformers, "AutoModelForImageTextToText"),
    )

    assert HFMultimodalLM.AUTO_MODEL_CLASS is expected


def test_bootstrapping():
    random.seed(42)
    arr = [random.random() for _ in range(1000)]
    expected = metrics.mean_stderr(arr)
    bootstrapped = metrics.bootstrap_stderr(metrics.mean, arr, iters=100000)

    assert bootstrapped == pytest.approx(expected, abs=1e-4)
