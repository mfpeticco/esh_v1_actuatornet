from actuator_network import train_actuator_network
from actuator_network import actuator_network_plot_predictions
from actuator_network import prepare_data
from glob import glob
import numpy as np

log_dir_root = "logs/"
log_dir = "esh_v2"
actuator_network_path = "models/esh_v2_new.pt"

load_pretrained_model = True

log_dirs = glob(f"{log_dir_root}{log_dir}/", recursive=True)


if len(log_dirs) == 0: raise FileNotFoundError(f"No log files found in {log_dir_root}{log_dir}/")

for log_dir in log_dirs:
    try:
        xs, ys, datavars, jointnames = prepare_data(log_dir[:11], log_dir[11:])
        if not load_pretrained_model:
            train_actuator_network(xs, ys, actuator_network_path)
        actuator_network_plot_predictions(xs, datavars, jointnames, actuator_network_path)
    except FileNotFoundError:
        print(f"Couldn't find log.pkl in {log_dir}")
    except EOFError:
        print(f"Incomplete log.pkl in {log_dir}")
