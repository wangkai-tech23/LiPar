from torch import nn
import torch


class FC_ELU(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        # fully connected layer followed with ELU activation
        super(FC_ELU, self).__init__(
            nn.Linear(in_channel, out_channel),
            nn.ELU(inplace=True)
        )



class CANet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1000, init_weights=False):
        super(CANet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = FC_ELU(32, 16)
        self.fc2 = FC_ELU(16, 4)
        self.fc3 = FC_ELU(4, 5)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.reshape(-1, 27, self.input_size)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
