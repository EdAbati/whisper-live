# This implementation was mostly ported from: https://github.com/Uberi/speech_recognition

import audioop
import collections
import io
import math
import threading
from collections.abc import Callable

from whisper_live.audio_source import AudioSource


class MonoAudioData:
    """
    Creates an instance that represents mono audio data.

    Args:
        frame_data: The raw audio data as a sequence of bytes representing audio samples.
            This is the frame data structure used by the PCM WAV format.
        sample_rate: The sample rate of the audio data in Hertz.
        sample_width: The width of each sample, in bytes.
            Each group of `sample_width` bytes represents a single audio sample.
    """

    def __init__(self, frame_data: bytes, sample_rate: int, sample_width: int):
        self._max_sample_width = 4

        if sample_rate <= 0:
            msg = "Sample rate must be a positive integer"
            raise ValueError(msg)

        if not isinstance(sample_width, int) or not (1 <= sample_width <= self._max_sample_width):
            msg = "Sample width must be an integer between 1 and 4 inclusive"
            raise ValueError(msg)
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = sample_width

    def get_raw_data(self, convert_rate: int | None = None, convert_width=None):
        """
        Returns a byte string representing the raw frame data for the audio.

        Args:
            convert_rate: The sample rate to convert to.
            convert_width: The sample width to convert to.


        Writing these bytes directly to a file results in a valid `RAW/PCM audio file <https://en.wikipedia.org/wiki/Raw_audio_format>`__.
        """

        if convert_rate is not None and convert_rate <= 0:
            msg = "Sample rate to convert to must be a positive integer"
            raise ValueError(msg)
        if convert_width is not None and not (1 <= convert_width <= self._max_sample_width):
            msg = "Sample width to convert to must be between 1 and 4 inclusive"
            raise ValueError(msg)

        raw_data = self.frame_data

        # make sure unsigned 8-bit audio (which uses unsigned samples) is handled like
        # higher sample width audio (which uses signed samples)
        if self.sample_width == 1:
            # subtract 128 from every sample to make them act like signed samples
            raw_data = audioop.bias(raw_data, 1, -128)

        # resample audio at the desired rate if specified
        if convert_rate is not None and self.sample_rate != convert_rate:
            raw_data, _ = audioop.ratecv(
                raw_data,
                self.sample_width,
                1,
                self.sample_rate,
                convert_rate,
                None,
            )

        # convert samples to desired sample width if specified
        if convert_width is not None and self.sample_width != convert_width:
            raw_data = audioop.lin2lin(raw_data, self.sample_width, convert_width)

            # if the output is 8-bit audio with unsigned samples,
            # convert the samples we've been treating as signed to unsigned again
            if convert_width == 1:
                # add 128 to every sample to make them act like unsigned samples again
                raw_data = audioop.bias(raw_data, 1, 128)

        return raw_data


def _raise_for_unentered_source(source: AudioSource):
    if source.stream is None:
        msg = "Audio source must be entered before recording. Please use `with source` to enter the source."
        raise ValueError(msg)


StopperFunc = Callable[[bool], None]


class AudioRecorder:
    def __init__(
        self,
        energy_threshold: int = 300,
        dynamic_energy_threshold: bool = True,
        dynamic_energy_adjustment_damping: float = 0.15,
        dynamic_energy_ratio: float = 1.5,
        pause_threshold: float = 0.8,
        operation_timeout: float | None = None,
        phrase_threshold: float = 0.3,
        non_speaking_duration: float = 0.5,
    ):
        """
        A Recorder class to record audio from audio sources.

        Args:
            energy_threshold: Minimum audio energy to consider for recording.
            dynamic_energy_threshold: TODO
            dynamic_energy_adjustment_damping: TODO
            dynamic_energy_ratio: TODO
            pause_threshold: Seconds of non-speaking audio before a phrase is considered complete.
            operation_timeout: Seconds after an internal operation (e.g., an API request) starts before it times out,
                or ``None`` for no timeout.
            phrase_threshold: Minimum seconds of speaking audio before we consider the speaking audio a phrase - values
                below this are ignored (for filtering out clicks and pops).
            non_speaking_duration: Seconds of non-speaking audio to keep on both sides of the recording.

        """

        if pause_threshold <= non_speaking_duration <= 0:
            msg = "pause_threshold must be greater than non_speaking_duration and greater than 0."
            raise ValueError(msg)

        self.energy_threshold = energy_threshold
        self.dynamic_energy_threshold = dynamic_energy_threshold
        self.dynamic_energy_adjustment_damping = dynamic_energy_adjustment_damping
        self.dynamic_energy_ratio = dynamic_energy_ratio
        self.pause_threshold = pause_threshold
        self.operation_timeout = operation_timeout
        self.phrase_threshold = phrase_threshold
        self.non_speaking_duration = non_speaking_duration

    def record(self, source: AudioSource, duration: float | None = None, offset: float | None = None) -> MonoAudioData:
        """
        Records up to `duration` seconds of audio from `source`.

        Optionally, one can start recording  at `offset` in seconds.

        Args:
            source: The audio source to record from.
            duration: The maximum number of seconds to record for.
                If `None`, it records until there is no more audio input.
            offset: The number of seconds from the beginning of the audio source to skip before starting to record.
        """

        _raise_for_unentered_source(source)

        with io.BytesIO() as audio_frames:
            elapsed_time = 0
            offset_time = 0
            offset_reached = False
            # loop for the total number of chunks needed
            while True:
                #
                if offset and not offset_reached:
                    offset_time += source.seconds_per_buffer
                    if offset_time > offset:
                        offset_reached = True

                buffer = source.stream.read(source.frames_per_buffer)
                if len(buffer) == 0:
                    break

                if offset_reached or not offset:
                    elapsed_time += source.seconds_per_buffer
                    if duration and elapsed_time > duration:
                        break

                    audio_frames.write(buffer)

            frame_data = audio_frames.getvalue()

        return MonoAudioData(frame_data=frame_data, sample_rate=source.sample_rate, sample_width=source.sample_width)

    def record_in_background(
        self,
        source: AudioSource,
        callback: Callable[[MonoAudioData], None],
    ) -> StopperFunc:
        """
        Spawns a thread to repeatedly record phrases from `source` (an `AudioSource` instance) into an `AudioData`
        instance and call `callback` with that `AudioData` instance as soon as each phrase are detected.

        Returns a function object that, when called, requests that the background listener thread stops.
        The background thread is a daemon and will not stop the program from exiting if there are no other non-daemon
        threads. The function accepts one parameter, `wait_for_stop`: if truthy, the function will wait for the
        background listener to stop before returning, otherwise it will return immediately and the background listener
        thread might still be running for a second or two afterwards. Additionally, if you are using a truthy value for
        `wait_for_stop`, you must call the function from the same thread you originally called `listen_in_background`
        from.

        Args:
            source: The audio source to record from.
            callback: The function to call when
        """
        running = [True]

        def threaded_record():
            with source as s:
                while running[0]:
                    try:
                        # Record for 0.5 second, then check again if the stop function has been called
                        audio = self.record(s, duration=0.5)
                    except TimeoutError:
                        # recording timed out, try again
                        pass
                    else:
                        if running[0]:
                            callback(audio)

        def stopper(wait_for_stop: bool = True):
            running[0] = False
            if wait_for_stop:
                recorder_thread.join()  # block until the background thread is done, which can take around 1 second

        recorder_thread = threading.Thread(target=threaded_record, daemon=True)
        recorder_thread.start()
        return stopper

    def _get_dynamic_adjusted_energy_threshold(self, audio_signal_energy: int, seconds_per_buffer: float) -> float:
        """
        Dynamically adjust the energy threshold using asymmetric weighted average
        """
        # account for different chunk sizes and rates
        damping = self.dynamic_energy_adjustment_damping**seconds_per_buffer
        target_energy = audio_signal_energy * self.dynamic_energy_ratio
        return self.energy_threshold * damping + target_energy * (1 - damping)

    def adjust_for_ambient_noise(self, source: AudioSource, duration: int = 1):
        """
        Adjusts the energy threshold dynamically using audio from `source` to account for ambient noise.

        Intended to calibrate the energy threshold with the ambient energy level.
        It should be used on periods of audio without speech - will stop early if any speech is detected.

        The `duration` parameter is the maximum number of seconds that it will dynamically adjust the threshold
        for before returning. This value should be at least 0.5 in order to get a representative sample of the ambient
        noise.
        """
        _raise_for_unentered_source(source)

        elapsed_time = 0
        # adjust energy threshold until a phrase starts
        while True:
            elapsed_time += source.seconds_per_buffer
            if elapsed_time > duration:
                break
            buffer = source.stream.read(source.frames_per_buffer)
            audio_signal_energy = audioop.rms(buffer, source.sample_width)
            # TODO why?
            self.energy_threshold = self._get_dynamic_adjusted_energy_threshold(
                audio_signal_energy=audio_signal_energy, seconds_per_buffer=source.seconds_per_buffer
            )

    def listen(self, source: AudioSource, timeout: float | None = None, phrase_time_limit: float | None = None):
        """
        Records a single phrase from `source` into an `AudioData` instance.

        This is done by waiting until the audio has an energy above `recognizer_instance.energy_threshold` (the user has
        started speaking), and then recording until it encounters `recognizer_instance.pause_threshold` seconds of
        non-speaking or there is no more audio input. The ending silence is not included.

        The `timeout` parameter is the maximum number of seconds that this will wait for a phrase to start before giving
        up and throwing an `speech_recognition.WaitTimeoutError` exception. If `timeout` is `None`, there will be
        no wait timeout.

        The `phrase_time_limit` parameter is the maximum number of seconds that this will allow a phrase to
        continue before stopping and returning the part of the phrase processed before the time limit was reached.
        The resulting audio will be the phrase cut off at the time limit.
        If `phrase_timeout` is `None`, there will be no phrase time limit.

        This operation will always complete within `timeout + phrase_timeout` seconds if both are numbers,
        either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
        """
        _raise_for_unentered_source(source)

        seconds_per_buffer = source.seconds_per_buffer
        # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))
        # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer))
        # maximum number of buffers of non-speaking audio to retain before and after a phrase
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer))
        # TODO: add logger debug for these values

        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0  # number of seconds of audio read
        buffer = b""  # an empty buffer means that the stream has ended and there is no data left to read
        while True:
            frames = collections.deque()

            # store audio input until the phrase starts
            # TODO: consider deleting or making this optional
            while True:
                # handle waiting too long for phrase by raising an exception
                elapsed_time += seconds_per_buffer
                if timeout and elapsed_time > timeout:
                    msg = "listening timed out while waiting for phrase to start"
                    raise TimeoutError(msg)

                buffer = source.stream.read(source.frames_per_buffer)

                if len(buffer) == 0:
                    # reached end of the stream
                    break

                frames.append(buffer)

                # ensure we only keep the needed amount of non-speaking buffers
                if len(frames) > non_speaking_buffer_count:
                    frames.popleft()

                # detect whether speaking has started on audio input
                audio_signal_energy = audioop.rms(buffer, source.sample_width)
                if audio_signal_energy > self.energy_threshold:
                    break

                # dynamically adjust the energy threshold using asymmetric weighted average
                if self.dynamic_energy_threshold:
                    # TODO: why?
                    self.energy_threshold = self._get_dynamic_adjusted_energy_threshold(
                        audio_signal_energy=audio_signal_energy, seconds_per_buffer=source.seconds_per_buffer
                    )

            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            phrase_start_time = elapsed_time
            while True:
                # handle phrase being too long by cutting off the audio
                elapsed_time += seconds_per_buffer
                if phrase_time_limit and elapsed_time - phrase_start_time > phrase_time_limit:
                    break

                buffer = source.stream.read(source.frames_per_buffer)
                if len(buffer) == 0:
                    break  # reached end of the stream
                frames.append(buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                audio_signal_energy = audioop.rms(buffer, source.sample_width)
                if audio_signal_energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:  # end of the phrase
                    break

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0:
                break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data
        for _ in range(pause_count - non_speaking_buffer_count):
            # remove extra non-speaking frames at the end
            frames.pop()
        frame_data = b"".join(frames)

        return MonoAudioData(frame_data, source.sample_rate, source.sample_width)

    def listen_in_background(
        self, source: AudioSource, callback: Callable[[MonoAudioData], None], phrase_time_limit: float | None = None
    ):
        """
        Spawns a thread to repeatedly record phrases from `source` into a `MonoAudioData`
        instance and call `callback` with that `MonoAudioData` instance as soon as each phrase are detected.

        Returns a function object that, when called, requests that the background listener thread stops.
        The background thread is a daemon and will not stop the program from exiting if there are no other non-daemon
        threads. The function accepts one parameter, `wait_for_stop`: if truthy, the function will wait for the
        background listener to stop before returning, otherwise it will return immediately and the background listener
        thread might still be running for a second or two afterwards. Additionally, if you are using a truthy value for
        `wait_for_stop`, you must call the function from the same thread you originally called `listen_in_background`
        from.

        Args:
            source: The audio source to record from.
            callback: The function to call when
        """
        running = [True]

        def threaded_listen():
            with source as s:
                while running[0]:
                    try:  # listen for 1 second, then check again if the stop function has been called
                        audio = self.listen(s, 1, phrase_time_limit)
                    except TimeoutError:  # listening timed out, just try again
                        pass
                    else:
                        if running[0]:
                            callback(audio)

        def stopper(wait_for_stop: bool = True):
            running[0] = False
            if wait_for_stop:
                listener_thread.join()  # block until the background thread is done, which can take around 1 second

        listener_thread = threading.Thread(target=threaded_listen, daemon=True)
        listener_thread.start()
        return stopper
