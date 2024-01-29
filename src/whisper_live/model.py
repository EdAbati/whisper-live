import abc
import time
from typing import Literal

import torch
from transformers import pipeline

from whisper_live.logging_utils import logger


class Model(abc.ABC):
    @property
    @abc.abstractmethod
    def sampling_rate(self): ...

    def transcribe(self, audio_array): ...


class MockModel(Model):
    @property
    def sampling_rate(self):
        return 16000

    def transcribe(self, audio_array):
        _ = audio_array
        return "mock transcription"


class HuggingFaceModel(Model):
    def __init__(
        self,
        model_name: str,
        device_id: str,
        torch_dtype: torch.dtype = torch.float16,
        use_flash_attention_2: bool = False,
        chunk_length_s: int = 30,
        batch_size: int = 24,
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str | None = None,
        timestamp: Literal["chunk", "word"] = "word",
    ) -> None:
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch_dtype,
            device="mps" if device_id == "mps" else f"cuda:{device_id}",
            model_kwargs=(
                {"attn_implementation": "flash_attention_2"}
                if use_flash_attention_2
                else {"attn_implementation": "sdpa"}
            ),
        )
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self.task = task
        self._timestamp = timestamp
        self.language = language
        self._timestamp = timestamp
        self.return_timestamps = "word" if timestamp == "word" else True

    @property
    def sampling_rate(self):
        return self.pipe.feature_extractor.sampling_rate

    def transcribe(self, audio_array):
        start_time = time.perf_counter()
        outputs = self.pipe(
            audio_array,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            generate_kwargs={"task": self.task, "language": self.language},
            return_timestamps=self.return_timestamps,
        )
        duration_s = time.perf_counter() - start_time
        logger.debug(f"Model transcription time: {duration_s:.3f}s")
        text = outputs["text"].strip()
        return text
