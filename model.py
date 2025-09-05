import pytorch_lightning as pl
import torch
import torch.nn as nn

class CrackImputer(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1) # Output a single value (crack width)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        lstm_out, _ = self.lstm(x)
        # Select the output from the last timestep for prediction
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction

    def training_step(self, batch, batch_idx):
        # A batch will contain a window of data and the value to predict
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y) # Mean Squared Error is a good loss function for regression
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer