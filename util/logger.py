import logging
import os
from datetime import datetime


def setup_logger(name='training', log_dir='logs', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'train.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name='training'):
    return logging.getLogger(name)
