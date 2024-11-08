import logging
import os
from datetime import datetime

def setup_logging():
    """
    配置日志记录器
    """
    # 清除之前的所有处理器，避免重复日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建logs目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 生成日志文件名（包含时间戳）
    log_filename = f"logs/video_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(request_id)s] %(message)s')
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 创建一个空的日志上下文字典
    logging.request_id = ''