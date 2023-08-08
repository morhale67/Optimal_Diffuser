import torch
import wandb
from Training import train
from Training import train_local
from Params import get_run_parameters
from LogFunctions import print_and_log_message
from LogFunctions import print_run_info_to_log
from LogFunctions import make_folder

# wandb.login(key='8aec627a04644fcae0f7f72d71bb7c0baa593ac6')
#
# sweep_id = 'ks5k7jhc'
# wandb.agent(sweep_id, train, project='Optimal Diffuser', count=20)

# local - without wandb
p = get_run_parameters()
folder_path = make_folder('GEN', p)
log_path = print_run_info_to_log(p)
print_and_log_message(f'learning rate: {p["lr"]}', log_path)

train_local(p, log_path)
