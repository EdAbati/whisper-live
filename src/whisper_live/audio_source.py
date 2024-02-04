# This implementation was mostly ported from: https://github.com/Uberi/speech_recognition

import abc
import audioop
import wave

import pyaudio

MONO_CHANNELS = 1
STEREO_CHANNELS = 2


class _AudioSourceStream(abc.ABC):
    @abc.abstractmethod
    def read(self, num_frames: int) -> bytes: ...


class AudioSource(abc.ABC):
    def __init__(self, sample_rate: int, sample_width: int, frames_per_buffer: int):
        """

        Args:
            sample_rate: Sampling rate to use.
            sample_width: Size of each sample.
            frames_per_buffer: Number of samples to read at a time.
        """
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.frames_per_buffer = frames_per_buffer

        self.audio = None
        self.stream: _AudioSourceStream | None = None

    @property
    def seconds_per_buffer(self) -> float:
        return float(self.frames_per_buffer) / self.sample_rate

    @abc.abstractmethod
    def __enter__(self): ...

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback): ...


class AudioFile(AudioSource):
    """
    Creates a new `AudioFile` instance given a WAV audio file.

    Note that functions that read from the audio (such as `recognizer_instance.record` or `recognizer_instance.listen`)
    will move ahead in the stream. For example, if you execute
    `recognizer_instance.record(audiofile_instance, duration=10)` twice, the first time it will return the first 10
    seconds of audio, and the second time it will return the 10 seconds of audio right after that.
    This is always reset to the beginning when entering an `AudioFile` context.

    WAV files must be in PCM/LPCM format; WAVE_FORMAT_EXTENSIBLE and compressed WAV are not supported and may result
    in undefined behaviour.
    """

    def __init__(self, file_path: str):
        if not isinstance(file_path, str):
            msg = "`file_path` must be a string"
            raise TypeError(msg)

        self.file_path = file_path
        with wave.open(self.file_path, "rb") as audio_file:
            if not MONO_CHANNELS <= audio_file.getnchannels() <= STEREO_CHANNELS:
                msg = f"Audio must be mono or stereo Got {audio_file.getnchannels()} channels"
                raise ValueError(msg)
            sample_width = audio_file.getsampwidth()
            sample_rate = audio_file.getframerate()
            self.frame_count = audio_file.getnframes()

        super().__init__(sample_rate=sample_rate, sample_width=sample_width, frames_per_buffer=4096)

        self.duration = self.frame_count / float(self.sample_rate)
        # RIFF WAV is a little-endian format
        # (most `audioop` operations assume that the frames are stored in little-endian form)
        self.little_endian = True

    def __enter__(self):
        if self.stream is not None:
            msg = "This audio source is already inside a context manager"
            raise RuntimeError(msg)

        self.audio = wave.open(self.file_path, "rb")
        self.stream = _AudioFileStream(self.audio, self.little_endian)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.audio.close()
        self.stream = None
        self.duration = None


class _AudioFileStream(_AudioSourceStream):
    def __init__(self, audio_reader: wave.Wave_read, little_endian: bool):
        self.audio_reader = audio_reader
        self.little_endian = little_endian

    def read(self, num_frames: int = -1) -> bytes:
        n_frames = (
            self.audio_reader.getnframes() if num_frames == -1 else num_frames
        )  # TODO can be taken from AudioFile
        buffer = self.audio_reader.readframes(n_frames)

        sample_width = self.audio_reader.getsampwidth()  # TODO can be taken from AudioFile
        if not self.little_endian:
            msg = "big endian audio files are not supported yet"
            raise NotImplementedError(msg)

        # Convert stereo audio to mono
        if self.audio_reader.getnchannels() == STEREO_CHANNELS:
            buffer = audioop.tomono(buffer, sample_width, 1, 1)
        return buffer


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


def _default_sample_rate_device(audio: pyaudio.PyAudio, device_id: int) -> int:
    """Return the default sample rate for the given hardware device."""
    device_info = audio.default_input_device_info if device_id is None else audio.get_device_info_by_index(device_id)
    default_sample_rate = device_info["defaultSampleRate"]
    if not isinstance(default_sample_rate, float | int) or default_sample_rate <= 0:
        msg = f"Invalid device info returned from PyAudio: {device_info}"
        raise ValueError(msg)
    return int(default_sample_rate)


class Microphone(AudioSource):
    """
    Represents a physical microphone on the computer.

    Args:
        device_index : Index of the microphone device to use. The default is to use the default microphone.
            A device index is an integer between 0 and `pyaudio.get_device_count() - 1`.
            See the [PyAudio documentation](https://people.csail.mit.edu/hubert/pyaudio/docs/) for more details.
        sample_rate : Sampling rate to use. The default is to use the device's default sampling rate
            (typically 16000 Hz).
            Higher `sample_rate` values result in better audio quality, but also more bandwidth
            (and therefore, slower processing). Additionally, some CPUs can't keep up if this value is too high.
        frames_per_buffer : Number of samples to read at a time from the microphone. The default is 1024.
            Higher `frames_per_buffer` values help avoid triggering on rapidly changing ambient noise, but also
            makes detection less sensitive. This value, generally, should be left at its default.

    The microphone audio is recorded in chunks of `frames_per_buffer` samples, at a rate of `sample_rate` samples per
    second (Hertz).
    """

    def __init__(self, device_index: int | None = None, sample_rate: int | None = None, frames_per_buffer: int = 1024):
        if device_index is not None and not isinstance(device_index, int):
            msg = "Device index must be None or an integer"
            raise ValueError(msg)

        if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
            msg = "Sample rate must be None or a positive integer"
            raise ValueError(msg)

        if not isinstance(frames_per_buffer, int) or frames_per_buffer <= 0:
            msg = "Chunk size must be a positive integer"
            raise ValueError(msg)

        audio = pyaudio.PyAudio()
        try:
            _raise_if_audio_device_invalid(audio, device_index)
            self.sample_rate = sample_rate or _default_sample_rate_device(audio, device_index)
        # TODO: check for exceptions?
        finally:
            audio.terminate()

        self.device_index = device_index
        # 16-bit int sampling
        self.format = pyaudio.paInt16

        super().__init__(
            sample_rate=sample_rate,
            sample_width=pyaudio.get_sample_size(self.format),
            frames_per_buffer=frames_per_buffer,
        )

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
                frames_per_buffer=self.frames_per_buffer,
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


class _MicrophoneStream(_AudioSourceStream):
    def __init__(self, pyaudio_stream: pyaudio.Stream):
        """
        Initializes an object that reads audio data from a microphone stream.

        Args:
            pyaudio_stream: The PyAudio stream object.
        """
        self.pyaudio_stream = pyaudio_stream

    def read(self, num_frames: int) -> bytes:
        """
        Reads audio data from the microphone stream.

        Args:
            num_frames: The number of frames to read.

        Returns:
            The audio data read as bytes.
        """
        return self.pyaudio_stream.read(num_frames=num_frames, exception_on_overflow=False)

    def close(self):
        """Closes the microphone stream."""
        try:
            # sometimes, if the stream isn't stopped, closing the stream throws an exception
            if not self.pyaudio_stream.is_stopped():
                self.pyaudio_stream.stop_stream()
        finally:
            self.pyaudio_stream.close()


def get_system_microphone(default_microphone: str | None = "pulse", sample_rate: int = 16000) -> Microphone:
    """Get the specified system microphone if available."""
    import sys

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if "linux" in sys.platform:
        mic_name = default_microphone
        if not mic_name or mic_name == "list":
            mic_names = "\n".join(f"- {n}" for n in list_microphone_names())
            err_msg = f"No microphone selected. Available microphone devices are:\n{mic_names}"
            raise ValueError(err_msg)
        else:
            for index, name in list_microphone_names():
                if mic_name in name:
                    return Microphone(sample_rate=sample_rate, device_index=index)
    return Microphone(sample_rate=sample_rate)
