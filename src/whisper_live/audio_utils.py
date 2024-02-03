from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue

import numpy as np
import speech_recognition as sr


def get_speech_recognizer(energy_threshold: int = 300) -> sr.Recognizer:
    """Set up a speech recognizer with a custom energy threshold."""
    # We use SpeechRecognizer to record our audio because it has a nice feature where
    # it can detect when speech ends.
    speech_recognizer = sr.Recognizer()
    speech_recognizer.energy_threshold = energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically
    # to a point where the SpeechRecognizer never stops recording.
    speech_recognizer.dynamic_energy_threshold = False
    return speech_recognizer


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
