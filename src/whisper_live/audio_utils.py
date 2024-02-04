from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue

import numpy as np
from scipy.io.wavfile import write


def to_audio_array(audio_data: bytes) -> np.ndarray:
    """
    Convert in-ram buffer to something the model can use directly without needing a temp file.

    Convert data from 16 bit wide integers to floating point with a width of 32 bits.
    Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
    """
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np


def get_all_audio_queue(data_queue: Queue) -> bytes:
    """Returns all audio in the queue."""
    audio_data = b"".join(data_queue.queue)
    data_queue.queue.clear()
    return audio_data


@dataclass
class AudioChunk:
    start_time: datetime
    end_time: datetime | None = None
    audio_array: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def duration(self) -> float | None:
        return None if self.end_time is None else self.end_time - self.start_time

    @property
    def is_complete(self) -> bool:
        return (self.end_time is not None) and (self.audio_array.size > 0)

    def update_array(self, new_audio: np.ndarray) -> None:
        self.audio_array = np.concatenate((self.audio_array, new_audio))

    def save(self, file_name: str | None = None) -> None:
        """Save the audio array to a file."""
        if file_name is None:
            file_name = f"data/audio_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}.wav"
        write(file_name, 16000, self.audio_array)
