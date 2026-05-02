import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class SileroVAD:
    """
    Silero Voice Activity Detection.

    Note: Silero VAD requires fixed-length inputs:
    - 512 samples for 16000 Hz
    - 256 samples for 8000 Hz
    """
    def __init__(
        self, threshold: float = 0.5, silence_duration_ms: int = 400, sample_rate: int = 16000
    ):
        self.threshold = threshold
        self.silence_duration_ms = silence_duration_ms
        self.sample_rate = sample_rate
        self.silence_samples = int(silence_duration_ms * sample_rate / 1000)

        # Silero VAD requires specific input sizes
        self.window_size = 512 if sample_rate == 16000 else 256

        self.model = None
        self.get_speech_timestamps = None
        self._load_model()

        self.is_speaking: bool = False
        self.speech_start_sample: int = 0
        self.last_speech_sample: int = 0
        self.silence_counter: int = 0

        self._ever_had_speech: bool = False
        self._initial_silence_samples: int = 0
        self._initial_silence_reported: bool = False

        self._buffer: np.ndarray = np.array([], dtype=np.float32)

    def _load_model(self):
        try:
            loaded = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True  # 关键参数：跳过交互式 y/N 确认
            )
            self.model = loaded[0] if isinstance(loaded, (list, tuple)) else None
            if isinstance(loaded, (list, tuple)) and len(loaded) > 1:
                utils = loaded[1]
                if isinstance(utils, (list, tuple)) and len(utils) > 0:
                    self.get_speech_timestamps = utils[0]
            if self.model:
                self.model.eval()
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
        except Exception as e:
            print(f"Warning: Failed to load Silero VAD: {e}")
            self.model = None

    def reset(self):
        self.is_speaking = False
        self.speech_start_sample = 0
        self.last_speech_sample = 0
        self.silence_counter = 0
        self._ever_had_speech = False
        self._initial_silence_samples = 0
        self._initial_silence_reported = False
        self._buffer = np.array([], dtype=np.float32)

    def process(self, audio_chunk: np.ndarray, total_samples: int) -> Dict[str, Any]:
        """
        Process audio chunk for VAD.

        Args:
            audio_chunk: Audio samples (any length)
            total_samples: Total samples processed so far

        Returns:
            Dict with speech_started/speech_stopped flags
        """
        if self.model is None:
            return {"speech_started": False, "speech_stopped": False}

        result: Dict[str, Any] = {"speech_started": False, "speech_stopped": False}

        # Add to buffer
        self._buffer = np.concatenate([self._buffer, audio_chunk])

        # Process in windows of required size
        chunk_start = total_samples - len(audio_chunk)

        while len(self._buffer) >= self.window_size:
            window = self._buffer[: self.window_size]
            self._buffer = self._buffer[self.window_size :]

            window_result = self._process_window(window, chunk_start)

            # Merge results
            if window_result.get("speech_started"):
                result["speech_started"] = True
                result["audio_start_ms"] = window_result["audio_start_ms"]
            if window_result.get("speech_stopped"):
                result["speech_stopped"] = True
                result["audio_end_ms"] = window_result["audio_end_ms"]

            chunk_start += self.window_size

        return result

    def _process_window(self, window: np.ndarray, window_start: int) -> Dict[str, Any]:
        """Process a single VAD window."""
        result: Dict[str, Any] = {}

        audio_tensor = torch.from_numpy(window).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()

        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        window_end = window_start + len(window)

        if speech_prob > self.threshold:
            self._ever_had_speech = True
            self._initial_silence_samples = 0

            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_sample = window_start
                result["speech_started"] = True
                result["audio_start_ms"] = int(self.speech_start_sample / self.sample_rate * 1000)

            self.last_speech_sample = window_end
            self.silence_counter = 0
        else:
            if self.is_speaking:
                self.silence_counter += len(window)

                if self.silence_counter >= self.silence_samples:
                    self.is_speaking = False
                    result["speech_stopped"] = True
                    result["audio_end_ms"] = int(self.last_speech_sample / self.sample_rate * 1000)
            elif not self._ever_had_speech and not self._initial_silence_reported:
                self._initial_silence_samples += len(window)

                if self._initial_silence_samples >= self.silence_samples:
                    self._initial_silence_reported = True
                    result["speech_stopped"] = True
                    result["audio_end_ms"] = int(window_end / self.sample_rate * 1000)

        return result

    def force_stop(self, total_samples: int) -> Optional[Dict[str, Any]]:
        if self.is_speaking:
            self.is_speaking = False
            return {
                "speech_stopped": True,
                "audio_end_ms": int(self.last_speech_sample / self.sample_rate * 1000),
            }
        return None


class VADManager:
    def __init__(
        self,
        enabled: bool = True,
        threshold: float = 0.5,
        silence_duration_ms: int = 400,
        sample_rate: int = 16000,
    ):
        self.enabled = enabled
        self.vad = SileroVAD(threshold, silence_duration_ms, sample_rate) if enabled else None

    def reset(self):
        if self.vad:
            self.vad.reset()

    def process(self, audio_chunk: np.ndarray, total_samples: int) -> Dict[str, Any]:
        if not self.enabled or self.vad is None:
            return {"speech_started": False, "speech_stopped": False}
        return self.vad.process(audio_chunk, total_samples)

    def force_stop(self, total_samples: int) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.vad is None:
            return None
        return self.vad.force_stop(total_samples)

    def is_speaking(self) -> bool:
        if not self.enabled or self.vad is None:
            return False
        return self.vad.is_speaking
