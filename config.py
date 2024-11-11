import os
import torch
from pathlib import Path

class Config:
    # 项目根目录
    BASE_DIR = Path(__file__).parent

    # 日志配置
    LOG_DIR = BASE_DIR / 'logs'
    LOG_RESOURCES_DIR = LOG_DIR / 'resources'
    LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # 临时文件配置
    TEMP_DIR = BASE_DIR / 'temp'
    
    # 服务器配置
    HOST = "0.0.0.0"
    PORT = 8000
    
    # 代理配置
    PROXY = "http://127.0.0.1:7890"
    
    # 资源阈值
    CPU_THRESHOLD = 80.0
    MEMORY_THRESHOLD = 80.0
    GPU_THRESHOLD = 80.0
    MAX_CONCURRENT_TASKS = 4
    
    # 音频处理配置
    SEGMENT_LENGTH = 25  # 音频分段长度（秒）
    
    # 模型配置
    MODEL_DIR = BASE_DIR / 'models'
    MODEL_NAME = "openai/whisper-tiny"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 下载配置
    DOWNLOAD_CHUNK_SIZE = 2 * 1024 * 1024  # 每个分片2MB
    MAX_CONCURRENT_CHUNKS = 5  # 最大并发下载数
    DOWNLOAD_TIMEOUT = 1800  # 下载超时时间（秒）
    CONNECT_TIMEOUT = 30  # 连接超时时间（秒）
    READ_TIMEOUT = 300  # 读取超时时间（秒）
    
    # 文本处理模式：'full' 或 'punctuation'
    TEXT_PROCESS_MODE = 'full'

    @classmethod
    def setup(cls):
        """创建必要的目录"""
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.LOG_RESOURCES_DIR, exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)