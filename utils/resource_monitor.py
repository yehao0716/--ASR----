import psutil
import time
import threading
from datetime import datetime
import logging
import os
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
from config import Config

class ResourceMonitor:
    def __init__(self, log_interval=5):
        """
        初始化资源监控器
        :param log_interval: 日志记录间隔（秒）
        """
        self.log_interval = log_interval
        self.is_running = False
        self.monitor_thread = None
        
        # 确保目录存在
        Config.setup()
        
        # 设置日志
        self.setup_logger()
        
    def setup_logger(self):
        """设置资源监控日志"""
        self.logger = logging.getLogger('ResourceMonitor')
        self.logger.setLevel(logging.INFO)
        
        # 只创建文件处理器，不创建控制台处理器
        log_file = Config.LOG_RESOURCES_DIR / f'resource_usage_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8-sig')
        file_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 只添加文件处理器
        self.logger.addHandler(file_handler)
    
    def get_cpu_usage(self):
        """获取CPU使用情况"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        return {
            'overall': sum(cpu_percent) / len(cpu_percent),
            'per_cpu': cpu_percent
        }
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 ** 3),  # GB
            'used': memory.used / (1024 ** 3),    # GB
            'percent': memory.percent
        }
    
    def get_gpu_usage(self):
        """获取GPU使用情况"""
        if not GPU_AVAILABLE:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            return [{
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory': {
                    'total': gpu.memoryTotal,  # MB
                    'used': gpu.memoryUsed,    # MB
                    'free': gpu.memoryFree     # MB
                },
                'temperature': gpu.temperature
            } for gpu in gpus]
        except Exception:
            return None
    
    def get_disk_usage(self):
        """获取磁盘使用情况"""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total / (1024 ** 3),  # GB
            'used': disk.used / (1024 ** 3),    # GB
            'free': disk.free / (1024 ** 3),    # GB
            'percent': disk.percent
        }
    
    def monitor_resources(self):
        """监控资源使用情况"""
        while self.is_running:
            try:
                # 获取资源使用情况
                cpu_usage = self.get_cpu_usage()
                memory_usage = self.get_memory_usage()
                gpu_usage = self.get_gpu_usage()
                disk_usage = self.get_disk_usage()
                
                # 记录资源使用情况
                self.logger.info(f"CPU使用率: {cpu_usage['overall']:.1f}%")
                self.logger.info(f"内存使用: {memory_usage['used']:.1f}GB/{memory_usage['total']:.1f}GB ({memory_usage['percent']}%)")
                
                if gpu_usage:
                    for gpu in gpu_usage:
                        self.logger.info(
                            f"GPU {gpu['id']} ({gpu['name']}): "
                            f"负载 {gpu['load']:.1f}%, "
                            f"显存 {gpu['memory']['used']}MB/{gpu['memory']['total']}MB, "
                            f"温度 {gpu['temperature']}°C"
                        )
                
                self.logger.info(
                    f"磁盘使用: {disk_usage['used']:.1f}GB/{disk_usage['total']:.1f}GB "
                    f"({disk_usage['percent']}%)"
                )
                self.logger.info("-" * 50)
                
            except Exception as e:
                self.logger.error(f"监控出错: {str(e)}")
            
            time.sleep(self.log_interval)
    
    def start(self):
        """开始监控"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self.monitor_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("资源监控已启动")
    
    def stop(self):
        """停止监控"""
        if self.is_running:
            self.is_running = False
            if self.monitor_thread:
                self.monitor_thread.join()
            self.logger.info("资源监控已停止") 