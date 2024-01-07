# Whisper Live

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/whisper-live.svg)](https://pypi.org/project/whisper-live)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/whisper-live.svg)](https://pypi.org/project/whisper-live) -->

-----

**Table of Contents**

- [Description](#description)
- [Contributing](#contributing)
- [License](#license)
- [Inspirations and Related Work](#inspirations-and-related-work)

## Description

> [!WARNING]
> This project is still a work in progress! See [Issues](https://github.com/EdAbati/whisper-live/issues) for a list of known bugs and missing features.

`whisper-live` is a CLI tool for real-time audio transcription using Whisper-based models.

It currently loads models using the [HuggingFace transformers](https://github.com/huggingface/transformers) library.

### Implementation logic

The implementation is highly based on https://github.com/davabase/whisper_real_time. More advanced logic could be considered for future releases.

The process is as follows:
- The audio is recorded via the microphone using a background thread
- Every _N_ seconds, an audio chunk is created. _N_ is defined by the `--recording-duration` argument.
- The audio chunk is sent to the model for transcription.
- The transcription is printed to the console.

Note: audio chunks (and transcriptions) are not aware of previous chunks. Once a chunk is processed, its audio is discarded. Once a transcription is printed, its text is not considered anymore.


## Contributing

1. Clone this repo
1. Make sure you have `hatch` installed. If not, follow the [guide here](https://hatch.pypa.io/dev/install/).
1. Create a virtual environment `hatch env create` and activate it `hatch shell`
1. Install `pre-commit` hooks for linting: `pre-commit install`

> [!WARNING]
> No unit tests are currently implemented!


## License

`whisper-live` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Inspirations and Related Work

This project was inspired by:
- https://github.com/davabase/whisper_real_time
- https://github.com/ufal/whisper_streaming
- https://github.com/oliverguhr/wav2vec2-live
