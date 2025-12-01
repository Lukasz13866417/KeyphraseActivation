import torch, torch.nn as nn, torch.nn.functional as F

""" A simple feature extractor (small CNN + small RNN) 
connected to MLP that turns per-moment features in to per-moment scores,
which are then pooled over time to get a single score for the entire window.

In the future, I might make a larger feature extractor with similar structure, 
pre-train it on multiple tasks (connected to multiple MLP heads) and use that as a backbone for the main models.

This should speed up the actual on-demand training and might even reduce the amount of data needed to train the main model.
But will require significantly more compute and data to train the extractor itself"""


class LogSumExpPool(nn.Module):
    def __init__(self, tau=0.5): 
        super().__init__()
        self.tau = tau
    def forward(self, x):
        """Pool over time by taking the log of the sum of the exponentials of the scores.
        Tau determins how close this is to the mean (tau=0 -> mean, tau=1 -> max)."""
        return self.tau * torch.logsumexp(x / self.tau, dim=1)  

class TinyCRNN(nn.Module):
    def __init__(self, in_ch=1, n_mels=40, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    nn.ReLU(),
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) 
        self.gru = nn.GRU(input_size=64, hidden_size=hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(p=0.2)
        self.frame_head = nn.Linear(hidden, 1)
        self.temporal_pool = LogSumExpPool(tau=0.5)      

    def forward(self, x):                
        h = self.conv(x)                  
        h = self.freq_pool(h).squeeze(2) 
        h = h.permute(0,2,1)           
        y,_ = self.gru(h)              
        y = self.drop(y)                  
        logits_t = self.frame_head(y).squeeze(-1)   
        window_logit = self.temporal_pool(logits_t)
        return window_logit, logits_t # per-moment logits (just for debugging) and the pooled window logit