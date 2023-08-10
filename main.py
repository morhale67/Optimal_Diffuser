import torch
import wandb
from Training import train
from Training import train_local
from Params import get_run_parameters
from LogFunctions import print_and_log_message
from LogFunctions import print_run_info_to_log
from OutputHandler import make_folder

wandb.login(key='8aec627a04644fcae0f7f72d71bb7c0baa593ac6')

sweep_id = 'ks5k7jhc'
wandb.agent(sweep_id, train, project='Optimal Diffuser', count=20)

# # local - without wandb
# p = get_run_parameters()
# for bs in [4, 32, 64]:
#     p['batch_size'] = bs
#     for lr in [0.01, 0.001, 0.0001, 10**(-7)]:
#         p['lr'] = lr
#         for cr in [2, 5, 10, 50]:
#             p['cr'] = cr
#             folder_path = make_folder('GEN', p)
#             log_path = print_run_info_to_log(p, folder_path) 
#             print_and_log_message(f'learning rate: {p["lr"]}', log_path)
#             train_local(p, log_path, folder_path)
