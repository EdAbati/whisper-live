import pytest

from whisper_live.audio_source import AudioFile, Microphone

AUDIO_FILE_PATH = "tests/data/english.wav"


def test_audio_file_init():
    audio_file = AudioFile(AUDIO_FILE_PATH)
    assert audio_file.sample_width == 2
    assert audio_file.sample_rate == 44100
    assert audio_file.frame_count == 121052
    assert audio_file.chunk_size == 4096


def test_audio_file_context_manager():
    audio_file = AudioFile(AUDIO_FILE_PATH)
    with audio_file as a:
        assert a.stream is not None
        assert a.audio is not None

    assert audio_file.stream is None
    assert audio_file.duration is None


def test_audio_file_stream_read():
    audio_file = AudioFile(AUDIO_FILE_PATH)
    with audio_file as a:
        buffer = a.stream.read(5)
        assert buffer == b"\x00\x00\xff\xff\x01\x00\xff\xff\x00\x00"


@pytest.mark.parametrize(
    "device_index,sample_rate,chunk_size", [(None, None, 1024), (None, 16000, 1024), (1, 16000, 1024)]
)
def test_microphone_init(device_index, sample_rate, chunk_size):
    mic = Microphone(device_index, sample_rate, chunk_size)
    assert mic.device_index == device_index
    assert mic.sample_rate == sample_rate
    assert mic.chunk_size == chunk_size


@pytest.mark.parametrize(
    "device_index,sample_rate,chunk_size",
    [("None", None, 1024), (None, "16000", 1024), (None, -16000, 1024), (None, 16000, "1024"), (None, 16000, -1024)],
)
def test_microphone_init_error(device_index, sample_rate, chunk_size):
    with pytest.raises(ValueError):
        Microphone(device_index, sample_rate, chunk_size)
