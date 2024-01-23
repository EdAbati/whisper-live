import pytest

from whisper_live.microphone import Microphone


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
