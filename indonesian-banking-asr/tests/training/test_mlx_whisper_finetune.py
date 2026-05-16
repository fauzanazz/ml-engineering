import numpy as np
import soundfile as sf
import mlx.core as mx
import mlx.nn as nn

from indonesian_banking_asr.training.mlx_whisper_finetune import LoRALinear, _build_training_example, _learning_rate_for_step, _merge_lora_linear
from mlx_whisper.tokenizer import get_tokenizer


def test_build_training_example_returns_mel_inputs_and_targets(tmp_path):
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
    tokenizer = get_tokenizer(True, language="id", task="transcribe")

    mel, decoder_inputs, targets = _build_training_example(
        {"audio_path": str(audio_path), "text": "halo dunia"},
        80,
        tokenizer,
    )

    assert mel.shape == (1, 3000, 80)
    assert decoder_inputs.shape[0] == 1
    assert targets.shape == decoder_inputs.shape


def test_learning_rate_for_step_warms_up_linearly():
    assert _learning_rate_for_step(1e-6, 1, 4, 10, "constant") == 2.5e-7
    assert _learning_rate_for_step(1e-6, 4, 4, 10, "constant") == 1e-6
    assert _learning_rate_for_step(1e-6, 5, 4, 10, "constant") == 1e-6
    assert _learning_rate_for_step(1e-6, 1, 0, 10, "constant") == 1e-6


def test_learning_rate_for_step_decays_after_warmup():
    assert _learning_rate_for_step(1e-6, 1, 2, 6, "warmup_linear_decay") == 5e-7
    assert _learning_rate_for_step(1e-6, 2, 2, 6, "warmup_linear_decay") == 1e-6
    assert _learning_rate_for_step(1e-6, 4, 2, 6, "warmup_linear_decay") == 5e-7
    assert _learning_rate_for_step(1e-6, 6, 2, 6, "warmup_linear_decay") == 0.0

def test_lora_linear_preserves_base_shape():
    layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=4.0)

    output = layer(mx.ones((1, 4)))

    assert output.shape == (1, 3)
    assert layer.lora_a.weight.shape == (2, 4)
    assert layer.lora_b.weight.shape == (3, 2)

def test_merge_lora_linear_returns_base_linear():
    layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=4.0)

    merged = _merge_lora_linear(layer)

    assert isinstance(merged, nn.Linear)
    assert merged.weight.shape == (3, 4)
