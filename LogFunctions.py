import logging
import time
from datetime import datetime
import os


def print_run_info_to_log(p, folder_path='Logs/'):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    log_name = f"Log_{dt_string}.log"
    print(f'Name of log file: {log_name}')
    log_path = folder_path + '/' + log_name
    logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f"This is a summery of the run:")
    logger.info(f"Batch size for this run: {p['batch_size']}")
    logger.info(f"Size of original image: {p['img_dim']}")
    # logger.info(f"Number of patterns: {p['m_patterns']} which is {p['prc_patterns']}% of {p['pic_width']}^2")
    # logger.info(f'Number of gray levels in output image: {n_gray_levels}')
    # logger.info(f"Initial lr: {p['initial_lr']}, Division factor: {p['div_factor_lr']}, {p['num_dif_lr']} divisions with {p['n_epochs']} epochs for each")
    # # logger.info(f"lr_vector {lr_vector}, epochs_vector{epochs_vector}")
    # print_and_log_message(f"Number of samples in train is {p['num_train_samples']}", log_path)
    # print_and_log_message(f"Number of samples in test is {p['num_test_samples']}", log_path)
    logger.info(f'Compression ratio: {p["cr"]}')
    logger.info(f'epochs : {p["epochs"]}')
    logger.info(f'number of fully connected layers in model: {p["n_fc"]}')
    logger.info('***************************************************************************\n\n')
    return log_path


def print_and_log_message(message, log_path):
    logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    if isinstance(message, str):
        logger.warning(message)
        print(message)
    else:
        logger.exception(message)


def print_training_messages(epoch, train_loss, avg_psnr, start, log_path):
    end = time.time()
    print_and_log_message(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss:.6f}', log_path)
    print_and_log_message(f"Time for epoch {epoch + 1} : {round(end - start)} sec", log_path)
    print_and_log_message(f'Average PSNR for epoch {epoch + 1} on training set is {avg_psnr:.6f}', log_path)


def make_folder(net_name, p):
    folder_name = f"{net_name}_bs_{p['batch_size']}_cr_{p['cr']}"
    print(folder_name)
    folder_path = 'Results/' + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path
