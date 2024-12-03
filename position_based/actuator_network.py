import os
import pickle as pkl
from matplotlib import pyplot as plt
import time
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import shutil
from datetime import datetime

def get_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders)")
  else:
    device = torch.device("cpu")
    print("Using CPU")
  return device

device = get_device()
num_dofs = 9

class ActuatorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['joint_states'])

    def __getitem__(self, idx):
        return {k: v[idx] for k,v in self.data.items()}

class Act(nn.Module):
  def __init__(self, act, slope=0.05):
    super(Act, self).__init__()
    self.act = act
    self.slope = slope
    self.shift = torch.log(torch.tensor(2.0)).item()

  def forward(self, input):
    if self.act == "relu":
      return F.relu(input)
    elif self.act == "leaky_relu":
      return F.leaky_relu(input)
    elif self.act == "sp":
      return F.softplus(input, beta=1.)
    elif self.act == "leaky_sp":
      return F.softplus(input, beta=1.) - self.slope * F.relu(-input)
    elif self.act == "elu":
      return F.elu(input, alpha=1.)
    elif self.act == "leaky_elu":
      return F.elu(input, alpha=1.) - self.slope * F.relu(-input)
    elif self.act == "ssp":
      return F.softplus(input, beta=1.) - self.shift
    elif self.act == "leaky_ssp":
      return (
          F.softplus(input, beta=1.) -
          self.slope * F.relu(-input) -
          self.shift
      )
    elif self.act == "tanh":
      return torch.tanh(input)
    elif self.act == "leaky_tanh":
      return torch.tanh(input) + self.slope * input
    elif self.act == "swish":
      return torch.sigmoid(input) * input
    elif self.act == "softsign":
        return F.softsign(input)
    else:
      raise RuntimeError(f"Undefined activation called {self.act}")

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

class Unnormalize(nn.Module):
    def __init__(self, mean, std):
        super(Unnormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

def build_mlp(in_dim, units, layers, out_dim, act='relu', layer_norm=False, act_final=False, x_mean=None, x_std=None, y_mean=None, y_std=None):
    mods = []
    if x_mean is not None and x_std is not None:
        mods.append(Normalize(x_mean, x_std))
    mods += [nn.Linear(in_dim, units), Act(act)]
    for i in range(layers-1):
        mods += [nn.Linear(units, units), Act(act)]
    mods += [nn.Linear(units, out_dim)]
    if act_final:
        mods += [Act(act)]
    if layer_norm:
        mods += [nn.LayerNorm(out_dim)]
    if y_mean is not None and y_std is not None:
        mods.append(Unnormalize(y_mean, y_std))
    return nn.Sequential(*mods)

def train_actuator_network(xs, ys, actuator_network_path):
    print(xs.shape, ys.shape)
    xs.to(device)
    ys.to(device)

    # Calculate mean and std for normalization
    xs_mean, xs_std = xs.mean(dim=0), xs.std(dim=0)
    ys_mean, ys_std = ys.mean(dim=0), ys.std(dim=0)
    num_data = xs.shape[0]
    num_train = num_data // 5 * 4
    num_test = num_data - num_train

    dataset = ActuatorDataset({"joint_states": xs, "joint_states_next": ys})
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=64, shuffle=True)

    in_dims = xs.shape[1]
    out_dims = ys.shape[1]
    model = build_mlp(in_dim=in_dims, units=32, layers=2, out_dim=out_dims, act='softsign', x_mean=xs_mean, x_std=xs_std, y_mean=ys_mean, y_std=ys_std)
    lr = 5e-4
    opt = Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.0)

    epochs = 300

    model = model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        ct = 0
        for batch in train_loader:
            data = batch['joint_states'].to(device)
            y_pred = model(data)

            opt.zero_grad()

            y_label = batch['joint_states_next'].to(device)

            tau_est_loss = ((y_pred - y_label) ** 2).mean()
            loss = tau_est_loss

            loss.backward()
            opt.step()
            epoch_loss += loss.detach().cpu().numpy()
            ct += 1
        epoch_loss /= ct

        test_loss = 0
        mae = 0
        ct = 0
        if epoch % 1 == 0:
            with torch.no_grad():
                for batch in test_loader:
                    data = batch['joint_states'].to(device)
                    y_pred = model(data)

                    y_label = batch['joint_states_next'].to(device)

                    tau_est_loss = ((y_pred - y_label) ** 2).mean()
                    loss = tau_est_loss
                    test_mae = (y_pred - y_label).abs().mean()

                    test_loss += loss
                    mae += test_mae
                    ct += 1
                test_loss /= ct
                mae /= ct

            print(
                f'epoch: {epoch} | loss: {epoch_loss:.4f} | test loss: {test_loss:.4f} | mae: {mae:.4f}')

        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(actuator_network_path)  # Save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # backup_path = actuator_network_path.replace(".pt", f"_{timestamp}.pt")
        # shutil.copyfile(actuator_network_path, backup_path)
    return model







def prepare_data(log_dir_root, log_dir, step=2):
    log_path = log_dir_root + log_dir + "log.pkl"
    print(log_path)
    with open(log_path, 'rb') as file:
        jointnames, datas = pkl.load(file)

    if len(datas) < 1:
        return

    joint_position = np.zeros((len(datas), num_dofs))
    joint_velocity = np.zeros((len(datas), num_dofs))
    joint_torque_cmd = np.zeros((len(datas), num_dofs))
    joint_position_next = np.zeros((len(datas), num_dofs))
    joint_velocity_next = np.zeros((len(datas), num_dofs))

    for i in range(len(datas)):
      joint_position[i, :] = datas[i]["joint_pos"]
      joint_velocity[i, :] = datas[i]["joint_vel"]
      joint_torque_cmd[i, :] = datas[i]["joint_torque_cmd"]
      joint_position_next[i, :] = datas[i]["joint_pos_next"]
      joint_velocity_next[i, :] = datas[i]["joint_vel_next"]

    joint_position = torch.tensor(joint_position, dtype=torch.float)
    joint_velocity = torch.tensor(joint_velocity, dtype=torch.float)
    joint_torque_cmd = torch.tensor(joint_torque_cmd, dtype=torch.float)
    joint_position_next = torch.tensor(joint_position_next, dtype=torch.float)
    joint_velocity_next = torch.tensor(joint_velocity_next, dtype=torch.float)

    xs = []
    ys = []
    # all joints are equal
    for i in range(num_dofs):
        xs_joint = [
                # joint_position[2:-step+1, i:i+1],
                # joint_position[1:-step, i:i+1],
                joint_position[:-step-1, i:i+1],
                # joint_velocity[2:-step+1, i:i+1],
                # joint_velocity[1:-step, i:i+1],
                joint_velocity[:-step-1, i:i+1],
                # joint_torque_cmd[2:-step+1, i:i+1],
                # joint_torque_cmd[1:-step, i:i+1],
                joint_torque_cmd[:-step-1, i:i+1]]
        
        # xs_joint = [joint_position[2:-step+1, i:i+1],
        #         joint_velocity[2:-step+1, i:i+1]]

        ys_joint = [joint_position_next[step:-1, i:i+1],
                    joint_velocity_next[step:-1, i:i+1]]

        xs_joint = torch.cat(xs_joint, dim=1) # (n x 9)
        ys_joint = torch.cat(ys_joint, dim=1)# (n x 2)

        xs += [xs_joint]
        ys += [ys_joint]

    xs = torch.cat(xs, dim=1) # (n x 9*num_dofs)
    ys = torch.cat(ys, dim=1) # (n x 2*num_dofs)
    datavars = (joint_position, joint_velocity, joint_torque_cmd, joint_position_next, joint_velocity_next)

    return (xs, ys, datavars, jointnames)


def actuator_network_plot_predictions(xs, datavars, jointnames, actuator_network_path, step=2):
    model = torch.jit.load(actuator_network_path).to('cpu')

    joint_position, joint_velocity, _, joint_position_next, joint_velocity_next = datavars
    joint_state_nextpred = model(xs).detach()
    joint_position_nextpred, joint_velocity_nextpred = joint_state_nextpred[:, ::2], joint_state_nextpred[:, 1::2]
    plot_length = 300

    timesteps = np.array(range(len(joint_position_nextpred))) / 50.0
    timesteps = timesteps[1:plot_length]

    joint_position = joint_position[step:plot_length+step]
    joint_velocity = joint_velocity[step:plot_length+step]
    joint_position_next = joint_position_next[step:plot_length+step]
    joint_velocity_next = joint_velocity_next[step:plot_length+step]
    
    joint_position_nextpred = joint_position_nextpred[:plot_length]
    joint_velocity_nextpred = joint_velocity_nextpred[:plot_length]

    fig, axs = plt.subplots(5, 2, figsize=(14, 8))
    axs = np.array(axs).flatten()
    for i in range(num_dofs):
        axs[i].plot(timesteps, joint_position_next[1:, i], label="true position")
        axs[i].plot(timesteps, joint_position_nextpred[1:, i], linestyle='--', label="actuator model predicted position")
        axs[i].set_title(jointnames[i])

    # Create a single legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.92, bottom=0.05)  # Adjust as needed

    plt.show()


