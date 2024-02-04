import argparse
from datetime import UTC, datetime, timedelta
from queue import Queue
from sys import platform
from time import sleep
from typing import Literal

import torch

from whisper_live import audio_source, audio_utils, model
from whisper_live.logging_utils import get_log_level, logger
from whisper_live.recorder import AudioRecorder, MonoAudioData
from whisper_live.transcribe_utils import Sentence


def _parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=False,
        default="openai/whisper-large-v3",
        type=str,
        help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
    )
    parser.add_argument(
        "--energy-threshold",
        default=300,
        help="Energy level for mic to detect. (default: 300)",
        type=int,
    )
    parser.add_argument(
        "--recording-duration",
        default=2,
        help=(
            "How many seconds each recording chunk should be before being sent to the model. "
            "Longer recordings maybe be more accurate, but the transcription will be shown with a longer delay. "
            "(default: 2)"
        ),
        type=float,
    )
    parser.add_argument(
        "--language",
        required=False,
        type=str,
        # TODO: change to "english" maybe? This field should also be validated against possible values.
        default="italian",
        help='Language of the input audio. (default: "italian")',
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        default=24,
        help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
    )
    parser.add_argument(
        "--timestamp",
        required=False,
        type=str,
        default="chunk",
        choices=["chunk", "word"],
        help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
    )
    parser.add_argument(
        "--device-id",
        required=False,
        default="mps",
        type=str,
        help=(
            "Device ID for your GPU. Just pass the device number when using CUDA, "
            'or "mps" for Macs with Apple Silicon. (default: "mps")'
        ),
    )
    if "linux" in platform:
        parser.add_argument(
            "--default-microphone",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones. (default: pulse)",
            type=str,
        )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increases verbosity of logging. Can be used multiple times to increase verbosity further.",
        action="count",
        dest="loglevel",
        default=0,
    )
    args = parser.parse_args()
    return args


def main(
    model_name: str,
    language: str,
    energy_threshold: int,
    recording_duration: float,
    batch_size: int,
    timestamp: Literal["chunk", "word"],
    device_id: str,
    default_microphone: str | None,
) -> None:
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()

    # Load / Download model
    transcribe_model = model.HuggingFaceModel(
        model_name=model_name,
        device_id=device_id,
        torch_dtype=torch.float16,
        use_flash_attention_2=False,
        batch_size=batch_size,
        task="transcribe",
        language=language,
        timestamp=timestamp,
    )
    logger.info("✅ Model loaded.")
    logger.info(f"🔈 Audio chunks of minimum {recording_duration}s will be sent to the model.")
    logger.info("🛑 Press Ctrl+C to stop recording!")

    microphone = audio_source.get_system_microphone(
        default_microphone=default_microphone, sample_rate=transcribe_model.sampling_rate
    )
    audio_recorder = AudioRecorder(energy_threshold=energy_threshold)

    with microphone:
        audio_recorder.adjust_for_ambient_noise(source=microphone)

    def record_callback(audio: MonoAudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.

        It gets the raw bytes from the audio and pushes it into the thread safe queue.

        Args:
            audio: An AudioData containing the recorded bytes.
        """
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    audio_recorder.listen_in_background(
        source=microphone,
        callback=record_callback,
        # Maximum number of seconds that this will allow a phrase to continue before stopping and
        # returning the part of the phrase processed before the time limit was reached.
        # The resulting audio will be the phrase cut off at the time limit.
        phrase_time_limit=recording_duration,
    )

    # Cue the user that we're ready to go.
    print("\n🎤 Microphone is now listening...\n")  # noqa: T201

    current_audio_chunk = audio_utils.AudioChunk(start_time=datetime.now(tz=UTC))

    while True:
        try:
            now = datetime.now(tz=UTC)
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                # Get audio data from queue
                audio_data = audio_utils.get_all_audio_queue(data_queue)
                audio_np_array = audio_utils.to_audio_array(audio_data)

                # Store end time if we're over the recording time limit.
                if now - current_audio_chunk.start_time >= timedelta(seconds=recording_duration):
                    current_audio_chunk.end_time = now

                if current_audio_chunk.is_complete:
                    logger.debug(f"Transcribing chunk of length {current_audio_chunk.duration}s ...")
                    text = transcribe_model.transcribe(current_audio_chunk.audio_array)
                    sentence = Sentence(
                        start_time=current_audio_chunk.start_time, end_time=current_audio_chunk.end_time, text=text
                    )
                    # current_audio_chunk.save()
                    current_audio_chunk = audio_utils.AudioChunk(
                        audio_array=audio_np_array, start_time=datetime.now(tz=UTC)
                    )
                    print(sentence.text)  # noqa: T201
                else:
                    current_audio_chunk.update_array(audio_np_array)

                # Flush stdout
                print("", end="", flush=True)  # noqa: T201

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            current_audio_chunk.end_time = datetime.now(tz=UTC)
            if current_audio_chunk.is_complete:
                logger.warning("⚠️ Transcribing last chunk...")
                text = transcribe_model.transcribe(current_audio_chunk.audio_array)
                sentence = Sentence(
                    start_time=current_audio_chunk.start_time, end_time=current_audio_chunk.end_time, text=text
                )
                print(sentence.text)  # noqa: T201
            break


def main_cli() -> None:
    args = _parse_cli_args()
    logger.setLevel(level=get_log_level(args.loglevel))
    main(
        model_name=args.model_name,
        language=args.language,
        energy_threshold=args.energy_threshold,
        recording_duration=args.recording_duration,
        batch_size=args.batch_size,
        timestamp=args.timestamp,
        device_id=args.device_id,
        default_microphone=args.default_microphone if "linux" in platform else None,
    )
