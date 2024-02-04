from datetime import UTC, datetime, timedelta
from queue import Queue
from time import sleep
from typing import Literal

import click
import torch

from whisper_live import audio_source, audio_utils, model
from whisper_live.logging_utils import get_log_level, logger
from whisper_live.recorder import AudioRecorder, MonoAudioData
from whisper_live.transcribe_utils import Sentence


@click.group()
def main_cli():
    """Transcribe live audio from your microphone using Whisper."""
    pass


@main_cli.command()
@click.option(
    "--kind",
    default="all",
    show_default=True,
    help="Kind of audio devices to list.",
    type=click.Choice(["input", "output", "all"]),
)
def list_devices(kind: Literal["input", "output", "all"]):
    """List all available audio devices."""
    from sounddevice import query_devices

    if kind == "all":
        click.echo(query_devices())
    else:
        device = query_devices(kind=kind)
        click.echo(
            f"The active {kind.capitalize()} device is:\n"
            f"  {device['index']} {device['name']} "
            f"({device['max_input_channels']} in, {device['max_output_channels']} out)"
        )


@main_cli.command()
@click.option(
    "--model-name",
    default="openai/whisper-large-v3",
    show_default=True,
    type=str,
    help="Name of the HuggingFace pretrained model/checkpoint to use to transcribe.",
)
@click.option("--energy-threshold", default=300, show_default=True, help="Energy level for mic to detect.", type=int)
@click.option(
    "--recording-duration",
    default=2,
    show_default=True,
    help=(
        "How many seconds each recording chunk should be before being sent to the model. "
        "Longer recordings maybe be more accurate, but the transcription will be shown with a longer delay."
    ),
    type=float,
)
@click.option("--language", default="italian", show_default=True, help="Language of the input audio.", type=str)
@click.option(
    "--batch-size",
    default=24,
    show_default=True,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs.",
    type=int,
)
@click.option(
    "--timestamp",
    default="chunk",
    show_default=True,
    help="Whisper supports both chunked and word level timestamps.",
    type=str,
)
@click.option(
    "--device-id",
    default="mps",
    show_default=True,
    help='Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon.',
    type=str,
)
@click.option(
    "--microphone_id",
    default=None,
    help="Default microphone name for SpeechRecognition. Run this with 'list' to view available Microphones.",
    type=int,
)
@click.option(
    "-v",
    "--verbose",
    help="Increases verbosity of logging. Can be used multiple times to increase verbosity further.",
    count=True,
    default=0,
)
def transcribe(
    model_name: str,
    language: str,
    energy_threshold: int,
    recording_duration: float,
    batch_size: int,
    timestamp: Literal["chunk", "word"],
    device_id: str,
    microphone_id: str | None,
    verbose: int,
) -> None:
    """Transcribe live audio recorded from the system microphone."""
    logger.setLevel(level=get_log_level(verbose))
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
        microphone_index=microphone_id, sample_rate=transcribe_model.sampling_rate
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
