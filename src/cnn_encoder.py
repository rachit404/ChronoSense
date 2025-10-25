# cnn_encoder.py

from typing import List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_length: int = 256, embedding_dim: int = 128):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        """
        x shape: (batch_size, 1, sequence_length)
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # shape (batch, 64, 1)
        x = x.squeeze(-1)  # shape (batch, 64)
        x = self.fc(x)     # shape (batch, embedding_dim)
        return x


class CNNEmbeddingPipeline:
    def __init__(self, window_size: int = 256, embedding_dim: int = 128, batch_size: int = 64, device: str = None):
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeSeriesCNN(input_length=window_size, embedding_dim=embedding_dim).to(self.device)
        self.model.eval()  # For inference
        print(f"[INFO] 1D CNN embedding model initialized on {self.device}.")

    def _prepare_windows(self, windows: List[np.ndarray]) -> torch.Tensor:
        """
        Convert list of 1D arrays (time-series windows) into tensor of shape (batch, 1, window_size)
        """
        tensor_data = np.stack(windows)  # shape (num_windows, window_size)
        tensor_data = np.expand_dims(tensor_data, axis=1)  # add channel dim: (num_windows, 1, window_size)
        return torch.from_numpy(tensor_data).float().to(self.device)

    def embed_windows(self, windows: List[np.ndarray]) -> np.ndarray:
        """
        windows: List of 1D numpy arrays of length `window_size`
        Returns: np.ndarray of shape (num_windows, embedding_dim)
        """
        tensor_data = self._prepare_windows(windows)
        embeddings = []

        with torch.no_grad():
            loader = DataLoader(tensor_data, batch_size=self.batch_size)
            for batch in loader:
                batch_emb = self.model(batch)
                embeddings.append(batch_emb.cpu().numpy())

        embeddings = np.vstack(embeddings)
        print(f"[INFO] Generated embeddings: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    # Example test
    windows = [np.random.randn(256) for _ in range(10)]
    pipeline = CNNEmbeddingPipeline()
    embeddings = pipeline.embed_windows(windows)
    print(embeddings.shape)
