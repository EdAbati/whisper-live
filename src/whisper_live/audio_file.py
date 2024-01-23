import audioop
import wave

MONO_CHANNELS = 1
STEREO_CHANNELS = 2


class AudioFile:
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
            self.sample_width = audio_file.getsampwidth()
            self.sample_rate = audio_file.getframerate()
            self.frame_count = audio_file.getnframes()

        self.chunk = 4096
        self.duration = self.frame_count / float(self.sample_rate)
        # RIFF WAV is a little-endian format
        # (most `audioop` operations assume that the frames are stored in little-endian form)
        self.little_endian = True

        self.audio = None
        self.stream = None

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


class _AudioFileStream:
    def __init__(self, audio_reader: wave.Wave_read, little_endian: bool):
        self.audio_reader = audio_reader
        self.little_endian = little_endian

    def read(self, size: int = -1):
        n_frames = self.audio_reader.getnframes() if size == -1 else size
        buffer = self.audio_reader.readframes(n_frames)

        sample_width = self.audio_reader.getsampwidth()
        if not self.little_endian:
            msg = "big endian audio files are not supported yet"
            raise NotImplementedError(msg)

        # Convert stereo audio to mono
        if self.audio_reader.getnchannels() == STEREO_CHANNELS:
            buffer = audioop.tomono(buffer, sample_width, 1, 1)
        return buffer
