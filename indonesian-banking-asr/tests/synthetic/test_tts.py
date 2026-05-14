import wave

from indonesian_banking_asr.synthetic.rate_limit import RateLimiter
from indonesian_banking_asr.synthetic.tts import NinerouterTts, EdgeTts, SyntheticToneTts, build_audio_manifest_rows


def test_ninerouter_tts_downloads_mp3_then_converts_to_wav(tmp_path):
    requests = []

    def post_speech(url, payload, headers, timeout_seconds):
        requests.append((url, payload, headers, timeout_seconds))
        return b"mp3-bytes"

    def convert_to_wav(mp3_path, wav_path, sample_rate):
        assert mp3_path.read_bytes() == b"mp3-bytes"
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((0).to_bytes(2, "little", signed=True) * 2)

    output_path = tmp_path / "9router.wav"
    tts = NinerouterTts(
        base_url="https://tts.example.test",
        api_key="secret-key",
        voice_name="edge-tts/id-ID-ArdiNeural",
        sample_rate=16000,
        post_speech=post_speech,
        convert_to_wav=convert_to_wav,
    )

    tts.synthesize("Saya mau cek saldo.", output_path)

    assert requests == [
        (
            "https://tts.example.test/v1/audio/speech",
            {"model": "edge-tts/id-ID-ArdiNeural", "input": "Saya mau cek saldo."},
            {"Content-Type": "application/json", "Authorization": "Bearer secret-key"},
            60,
        )
    ]
    assert not (tmp_path / "9router.mp3").exists()
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 2


def test_edge_tts_writes_mp3_then_converts_to_wav(tmp_path):
    class StubCommunicate:
        def __init__(self, text, voice):
            self.text = text
            self.voice = voice

        async def save(self, path):
            assert self.text == "Saya mau cek saldo."
            assert self.voice == "id-ID-ArdiNeural"
            path.write_bytes(b"mp3-bytes")

    def convert_to_wav(mp3_path, wav_path, sample_rate):
        assert mp3_path.read_bytes() == b"mp3-bytes"
        assert sample_rate == 16000
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((0).to_bytes(2, "little", signed=True) * 2)

    output_path = tmp_path / "edge.wav"
    tts = EdgeTts(
        voice_name="id-ID-ArdiNeural",
        sample_rate=16000,
        communicate_factory=StubCommunicate,
        convert_to_wav=convert_to_wav,
    )

    tts.synthesize("Saya mau cek saldo.", output_path)

    assert not (tmp_path / "edge.mp3").exists()
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 2


def test_build_audio_manifest_rows_writes_wav_and_adds_audio_metadata(tmp_path):
    rows = [
        {
            "utterance_id": "utt-001",
            "text": "Saya mau cek saldo rekening 1234567890.",
        }
    ]
    audio_dir = tmp_path / "audio"

    audio_rows = build_audio_manifest_rows(
        rows,
        audio_dir=audio_dir,
        tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
    )

    audio_path = audio_dir / "utt-001.wav"
    assert audio_path.exists()
    assert audio_rows == [
        {
            "utterance_id": "utt-001",
            "text": "Saya mau cek saldo rekening 1234567890.",
            "audio_path": str(audio_path),
            "duration_sec": 0.25,
            "sample_rate": 8000,
            "tts_engine": "synthetic-tone",
        }
    ]


def test_build_audio_manifest_rows_skips_processed_rows_when_resuming(tmp_path):
    rows = [
        {"utterance_id": "utt-001", "text": "satu"},
        {"utterance_id": "utt-002", "text": "dua"},
    ]
    audio_dir = tmp_path / "audio"
    output_path = tmp_path / "audio_manifest.jsonl"
    output_path.write_text('{"utterance_id":"utt-001","audio_path":"old.wav"}\n')

    audio_rows = build_audio_manifest_rows(
        rows,
        audio_dir=audio_dir,
        tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
        processed_utterance_ids={"utt-001"},
    )

    assert [row["utterance_id"] for row in audio_rows] == ["utt-002"]
    assert not (audio_dir / "utt-001.wav").exists()
    assert (audio_dir / "utt-002.wav").exists()


def test_build_audio_manifest_rows_waits_before_each_tts_request(tmp_path):
    sleeps = []
    limiter = RateLimiter(seconds_per_request=2.5, sleep=sleeps.append)

    build_audio_manifest_rows(
        [
            {"utterance_id": "utt-001", "text": "satu"},
            {"utterance_id": "utt-002", "text": "dua"},
        ],
        audio_dir=tmp_path / "audio",
        tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
        rate_limiter=limiter,
    )

    assert sleeps == [0.0, 2.5]


def test_build_audio_manifest_rows_rejects_missing_text(tmp_path):
    rows = [{"utterance_id": "utt-001"}]

    try:
        build_audio_manifest_rows(
            rows,
            audio_dir=tmp_path / "audio",
            tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
        )
    except ValueError as error:
        assert str(error) == "row utt-001 missing text"
    else:
        raise AssertionError("expected missing text to fail")
