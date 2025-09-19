import pytorch_lightning as pl
import torch
import torch.nn as nn


# The LightningModule defines the model architecture, loss function, and training steps.
class CrackImputer(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        # PyTorch Lightning will save these as hyperparameters
        self.save_hyperparameters()
        
        # The LSTM layer expects input of shape (batch_size, seq_len, num_features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # The final fully-connected layer to produce the single-value prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Pass the input through the LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last timestep and pass it through the final layer
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y.unsqueeze(1))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer