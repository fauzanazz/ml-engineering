import numpy as np
import soundfile as sf

from indonesian_banking_asr.training.whisper_finetune import ManifestSpeechDataset


class FakeFeatureExtractor:
    def __call__(self, audio, *, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        import torch

        return type("FeatureResult", (), {"input_features": torch.zeros((1, 80, 3000))})()


class FakeTokenizer:
    def __call__(self, text, *, return_tensors):
        assert text == "halo dunia"
        assert return_tensors == "pt"
        import torch

        return type("TokenResult", (), {"input_ids": torch.tensor([[1, 2, 3]])})()


class FakeProcessor:
    feature_extractor = FakeFeatureExtractor()
    tokenizer = FakeTokenizer()


def test_manifest_speech_dataset_loads_audio_and_labels(tmp_path):
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
    dataset = ManifestSpeechDataset(
        [{"utterance_id": "utt-1", "audio_path": str(audio_path), "text": "halo dunia"}],
        FakeProcessor(),
    )

    item = dataset[0]

    assert tuple(item["input_features"].shape) == (80, 3000)
    assert item["labels"].tolist() == [1, 2, 3]
    assert item["utterance_id"] == "utt-1"
