import wandb
from main_training import train
from Params import get_run_parameters
from LogFunctions import print_and_log_message
from LogFunctions import print_run_info_to_log
from OutputHandler import make_folder
from Params import load_config_parameters



def main():
    wb_flag = True
    p = get_run_parameters()
    if wb_flag:
        p = load_config_parameters(p)
    folder_path = make_folder(p)
    log_path = print_run_info_to_log(p, folder_path)
    print_and_log_message(f'learning rate: {p["lr"]}', log_path)
    train(p, log_path, folder_path, wb_flag)


wandb.login(key='8aec627a04644fcae0f7f72d71bb7c0baa593ac6')

sweep_configuration = {
    "method": "bayes",
    "metric": {
        "goal": "minimize",
        "name": "Wall Time"
    },
    "parameters": {
        "batch_size": {
            "values": [2, 4]
        },
        "pic_width": {
            "values": [28, 32]
        },
        "z_dim": {
            "values": [32, 64, 128]
        },
        "weight_decay": {
            "values": [1e-7, 5e-7, 10e-7]
        },
        "TV_beta": {
            "values": [0.1, 0.5, 1.0, 10, 100]
        },
        "cr": {
            "values": [2, 5, 10]
        }
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="Optimal Diffuser")
wandb.agent(sweep_id, function=main, count=20)

