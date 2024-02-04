from whisper_live.audio_source import AudioFile
from whisper_live.recorder import AudioRecorder, MonoAudioData


def test_recognizer_attributes():
    recorder = AudioRecorder()

    assert recorder.energy_threshold == 300
    assert recorder.dynamic_energy_threshold
    assert recorder.dynamic_energy_adjustment_damping == 0.15
    assert recorder.dynamic_energy_ratio == 1.5
    assert recorder.pause_threshold == 0.8
    assert recorder.operation_timeout is None
    assert recorder.phrase_threshold == 0.3
    assert recorder.non_speaking_duration == 0.5


def test_recorder_record():
    recorder = AudioRecorder()
    with AudioFile("tests/data/english.wav") as audio_file:
        audio = recorder.record(audio_file)
    assert isinstance(audio, MonoAudioData)
    assert audio.get_raw_data().startswith(b"\x00\x00\xff\xff\x01\x00\xff\xff\x00\x00\x01")
    assert audio.get_raw_data().endswith(b"\x02\x00\xff\xff\x00\x00\x00\x00\xff\xff\x01\x00")
