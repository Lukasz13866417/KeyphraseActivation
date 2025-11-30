class SpectrogramConfig:
    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 40):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def to_dict(self):
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels
        }