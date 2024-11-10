import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# For FHN hyperparameters
t_max = 100
dt = 0.1

def fhn_equations(U, W, K):
  dU_dt = K * U * (U - 0.4) * (1 - U) - W
  dW_dt = 0.5 * (0.2 * U - 0.8 * W)
  return dU_dt, dW_dt

def euler_solver(u0, w0, K, t_max, dt):
    """Solve FHN model using Euler method"""
    t = np.arange(0, t_max, dt)
    u = np.zeros_like(t)
    w = np.zeros_like(t)
    u[0], w[0] = u0, w0
    for i in range(1, len(t)):
        du_dt, dw_dt = fhn_equations(u[i - 1], w[i - 1], K)
        u[i] = u[i - 1] + du_dt * dt
        w[i] = w[i - 1] + dw_dt * dt
    return t, u, w

def generate_fhn_data(num_samples):
    """Generate data for different initial conditions and K values"""
    data = []
    for i in range((num_samples)):
      u0 = np.random.uniform(.5,0.6)  # allowed to be vary 
      w0 = np.random.uniform(0,.1)   # allowed to be vary 
      K = np.random.uniform(2,2.5)  # allowed to be vary 
      t, u, w = euler_solver(u0, w0, K, t_max, dt)
      data.append({'t': t,'u0': u0, 'w0': w0, 'K': K, 'u': u, 'w': w})
    return data

def visualize_fhn_samples(training_data):
    # Randomly select four samples to plot
    num_samples = len(training_data)
    random_indices = np.random.choice(num_samples, size=4, replace=False)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

    # Flatten the array of axes for easy iteration
    axs = axs.flatten()

    # Loop through the randomly selected samples and plot their u and w values
    for i, idx in enumerate(random_indices):
        axs[i].plot(training_data[idx]['t'], training_data[idx]['u'], label='u', color='blue')
        axs[i].plot(training_data[idx]['t'], training_data[idx]['w'], label='w', color='orange', linestyle='dotted')
        
        # Set labels and title for each subplot
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Value')
        axs[i].set_title(f"FHN Model Sample: u0={training_data[idx]['u0']:.2f}, w0={training_data[idx]['w0']:.2f}, K={training_data[idx]['K']:.2f}")
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()
    plt.savefig('fhn_samples.png')

class FHN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FHN_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_u = nn.Linear(hidden_size, output_size)
        self.fc_w = nn.Linear(hidden_size, output_size)
        self.input_fc = nn.Linear(3, input_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, u0, w0, K):
        x = torch.cat((u0, w0, K), dim=1).unsqueeze(1)
        x = self.input_fc(x)
        u_list = []
        w_list = []

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        for _ in range(int(t_max/dt)):
            out, (hn, cn) = self.lstm(x, (h0, c0))
            x = out
            predicted_u = self.fc_u(out)
            predicted_w = self.fc_w(out)
            u_list.append(predicted_u)
            w_list.append(predicted_w)
        return torch.stack(u_list, dim=1).squeeze(), torch.stack(w_list, dim=1).squeeze()

def train_LSTM(model, criterion, optimizer, training_data, validation_data, num_epochs=100):
    train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for i, sample in enumerate(train_loader):
            # Get inputs
            input_u0 = sample['u0'].float().to(device).unsqueeze(1)
            input_w0 = sample['w0'].float().to(device).unsqueeze(1)
            input_K = sample['K'].float().to(device).unsqueeze(1)
            # Get targets
            target_u = sample['u'].float().to(device)
            target_w = sample['w'].float().to(device)


            optimizer.zero_grad()
            predicted_u, predicted_w = model(input_u0, input_w0, input_K)
            loss = criterion(predicted_u, target_u) + criterion(predicted_w, target_w)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        train_losses.append(training_loss / len(train_loader))

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                # Get inputs
                input_u0 = sample['u0'].float().to(device).unsqueeze(1)
                input_w0 = sample['w0'].float().to(device).unsqueeze(1)
                input_K = sample['K'].float().to(device).unsqueeze(1)
                # Get targets
                target_u = sample['u'].float().to(device)
                target_w = sample['w'].float().to(device)

                predicted_u, predicted_w = model(input_u0, input_w0, input_K)
                loss = criterion(predicted_u, target_u) + criterion(predicted_w, target_w)

                validation_loss += loss.item()
            val_losses.append(validation_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

if __name__ == '__main__':
    # Generate Training Data and Validation Data
    training_num_samples = 1000 
    validation_num_samples = 200
    training_data = generate_fhn_data(num_samples=training_num_samples)
    validation_data = generate_fhn_data(num_samples=validation_num_samples)

    # Construct LSTM Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 64
    hidden_size = 64
    num_layers = 2
    output_size = 1
    model = FHN_LSTM(input_size, hidden_size, num_layers, output_size).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the LSTM Model
    train_losses, val_losses = train_LSTM(model, criterion, optimizer, training_data, validation_data, num_epochs=100)




