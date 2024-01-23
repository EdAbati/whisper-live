import pyaudio


def list_microphone_names() -> list[str]:
    """
    Returns the names of all available microphones.

    For microphones where the name can't be retrieved, the list entry contains `None` instead.
    """
    audio = pyaudio.PyAudio()
    try:
        microphone_names = [audio.get_device_info_by_index(i).get("name") for i in range(audio.get_device_count())]
    finally:
        audio.terminate()
    return microphone_names


def _raise_if_audio_device_invalid(audio: pyaudio.PyAudio, device_id: int) -> None:
    """
    Raises a ValueError if the given device ID is invalid.
    """
    device_count = audio.get_device_count()
    if device_id is not None and not 0 <= device_id < device_count:
        msg = (
            f"Device index out of range. "
            f"({device_count} devices available; device index should be between 0 and {device_count - 1} inclusive)"
        )
        raise ValueError(msg)


def _default_sample_rate(audio: pyaudio.PyAudio, device_id: int) -> int:
    """Return the default sample rate for the given hardware device."""
    device_info = audio.default_input_device_info if device_id is None else audio.get_device_info_by_index(device_id)
    default_sample_rate = device_info["defaultSampleRate"]
    if not isinstance(default_sample_rate, float | int) or default_sample_rate <= 0:
        msg = f"Invalid device info returned from PyAudio: {device_info}"
        raise ValueError(msg)
    return int(default_sample_rate)


class Microphone:
    """
    Represents a physical microphone on the computer.

    Args:
        device_index : Index of the microphone device to use. The default is to use the default microphone.
            A device index is an integer between 0 and `pyaudio.get_device_count() - 1`.
            It represents an audio device such as a microphone or speaker.
            See the `PyAudio documentation <http://people.csail.mit.edu/hubert/pyaudio/docs/>`__ for more details.
        sample_rate : Sampling rate to use. The default is to use the device's default sampling rate
            (typically 16000 Hz). Higher `sample_rate` values result in better audio quality, but also more bandwidth
            (and therefore, slower recognition). Additionally, some CPUs, such as those in older Raspberry Pi models,
            can't keep up if this value is too high.
        chunk_size : Number of samples to read at a time from the microphone. The default is 1024.
            Higher `chunk_size` values help avoid triggering on rapidly changing ambient noise, but also makes
            detection less sensitive. This value, generally, should be left at its default.

    The microphone audio is recorded in chunks of `chunk_size` samples, at a rate of `sample_rate` samples per
    second (Hertz).
    """

    def __init__(self, device_index: int | None = None, sample_rate: int | None = None, chunk_size: int = 1024):
        if device_index is not None and not isinstance(device_index, int):
            msg = "Device index must be None or an integer"
            raise ValueError(msg)

        if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
            msg = "Sample rate must be None or a positive integer"
            raise ValueError(msg)

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            msg = "Chunk size must be a positive integer"
            raise ValueError(msg)

        audio = pyaudio.PyAudio()
        try:
            _raise_if_audio_device_invalid(audio, device_index)
            self.sample_rate = sample_rate or _default_sample_rate(audio, device_index)
        # TODO: check for exceptions?
        finally:
            audio.terminate()

        self.device_index = device_index
        # number of frames stored in each buffer
        # TODO: rename to frames_per_buffer
        self.chunk_size = chunk_size
        # 16-bit int sampling
        self.format = pyaudio.paInt16
        # size of each sample
        self.sample_width = pyaudio.get_sample_size(self.format)

        self.audio = None
        self.stream = None

    def __enter__(self):
        if self.stream is not None:
            msg = "This audio source is already inside a context manager"
            raise RuntimeError(msg)
        self.audio = pyaudio.PyAudio()
        try:
            pyaudio_stream = self.audio.open(
                input_device_index=self.device_index,
                channels=1,
                format=self.format,
                rate=self.sample_rate,
                frames_per_buffer=self.chunk_size,
                input=True,
            )
            self.stream = _MicrophoneStream(pyaudio_stream)
        # TODO: check for specific exceptions?
        except Exception:
            self.audio.terminate()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stream.close()
        finally:
            self.stream = None
            self.audio.terminate()


class _MicrophoneStream:
    def __init__(self, pyaudio_stream: pyaudio.Stream):
        self.pyaudio_stream = pyaudio_stream

    def read(self, num_frames: int):
        return self.pyaudio_stream.read(num_frames=num_frames, exception_on_overflow=False)

    def close(self):
        try:
            # sometimes, if the stream isn't stopped, closing the stream throws an exception
            if not self.pyaudio_stream.is_stopped():
                self.pyaudio_stream.stop_stream()
        finally:
            self.pyaudio_stream.close()
