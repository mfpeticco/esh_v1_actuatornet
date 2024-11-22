from actuator_network import train_actuator_network_and_plot_predictions
from glob import glob

log_dir_root = "logs/"
log_dir = "real_go1"

# Evaluates the existing actuator network by default
# load_pretrained_model = True
# actuator_network_path = "../../resources/actuator_nets/unitree_go1.pt"

# Uncomment these lines to train a new actuator network
load_pretrained_model = False
actuator_network_path = "models/unitree_go1_new.pt"


log_dirs = glob(f"{log_dir_root}{log_dir}/", recursive=True)

if len(log_dirs) == 0: raise FileNotFoundError(f"No log files found in {log_dir_root}{log_dir}/")

for log_dir in log_dirs:
    try:
        train_actuator_network_and_plot_predictions(log_dir[:11], log_dir[11:], actuator_network_path=actuator_network_path, load_pretrained_model=load_pretrained_model)
    except FileNotFoundError:
        print(f"Couldn't find log.pkl in {log_dir}")
    except EOFError:
        print(f"Incomplete log.pkl in {log_dir}")
