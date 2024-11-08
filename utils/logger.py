import logging
from datetime import datetime
from config import Config

def setup_logger():
    """设置日志配置"""
    # 确保日志目录存在
    Config.setup()
    
    logger = logging.getLogger('VideoASR')
    
    # 如果logger已经有handlers，说明已经初始化过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)  # 默认级别为 INFO
    
    # 创建控制台处理器（INFO级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建文件处理器（DEBUG级别，记录所有日志）
    log_file = Config.LOG_DIR / f'video_asr_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8-sig')
    file_handler.setLevel(logging.DEBUG)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt=Config.LOG_DATE_FORMAT
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger