"""
Written by KrishPro @ KP

filename: `microphone.py`
"""

import numpy as np
import pyaudio

class Microphone:
    """
    This class works with pyaudio at a very low level and can be used to control microphone.
    """

    def __init__(self, rate: int, channels: int, chunk_size: int):
        self.setup(rate, channels, chunk_size)
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size

    def setup(self, rate: int, channels: int, chunk_size: int, format=pyaudio.paInt16, input=True):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(format=format, channels=channels, rate=rate, input=input, frames_per_buffer=chunk_size)
            
    def read(self, seconds: int):
        audio = np.array([], dtype=np.int16)
        num_samples_required = int(self.rate * seconds)
       
        num_chunks_required = int(num_samples_required // self.chunk_size)
        if  num_samples_required % self.chunk_size != 0: num_chunks_required += 1
            
        for _ in range(num_chunks_required):
            chunk:np.ndarray = np.frombuffer(self.stream.read(self.chunk_size), dtype=np.int16)
            audio: np.ndarray = np.concatenate([audio, chunk], 0)

        audio = audio[-num_samples_required:]
        assert len(audio) == num_samples_required
        assert len(audio) / self.rate == seconds
        return audio

    def close(self):
        self.stream.stop_stream()
        self.stream.close()