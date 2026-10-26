"""Speech: transcription for voice search, the speech test set, and the
voice-robustness evaluation.

- ``audio``: WAV decoding and resampling to the 16 kHz mono float the
  recognizer takes.
- ``transcribe``: the speech-to-text backends behind the speech server.
- ``server``: the speech server the Go API calls for ``POST /v1/search/voice``.
- ``degrade``, ``testset``: the synthetic speech test set.
- ``wer``: word and named-entity error rates.
- ``evaluate``: speech-induced retrieval degradation, measured through the
  running API.
"""
