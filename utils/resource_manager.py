import psutil
import GPUtil
from dataclasses import dataclass
from typing import Optional
import asyncio
import logging

# 获取logger
logger = logging.getLogger('VideoASR')

@dataclass
class ResourceThreshold:
    """资源阈值配置"""
    max_cpu_percent: float = 80.0      # CPU使用率阈值
    max_memory_percent: float = 80.0    # 内存使用率阈值
    max_gpu_percent: float = 80.0       # GPU使用率阈值
    max_concurrent_tasks: int = 4       # 最大并发任务数

class ResourceManager:
    def __init__(self, threshold: ResourceThreshold = ResourceThreshold()):
        self.threshold = threshold
        self.current_tasks = 0
        self._lock = asyncio.Lock()
        
    async def _check_resources_internal(self) -> tuple[bool, str]:
        """
        内部资源检查方法，不使用锁
        """
        # 检查当前任务数
        if self.current_tasks >= self.threshold.max_concurrent_tasks:
            logger.warning(f"当前任务数({self.current_tasks})已达上限({self.threshold.max_concurrent_tasks})")
            return False, f"当前任务数({self.current_tasks})已达上限({self.threshold.max_concurrent_tasks})"
            
        # 在线程池中执行同步的资源检查
        def check_sync():
            # 检查CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            logger.info(f"当前CPU使用率: {cpu_percent:.1f}%")
            if cpu_percent > self.threshold.max_cpu_percent:
                logger.warning(f"CPU使用率({cpu_percent:.1f}%)超过阈值({self.threshold.max_cpu_percent}%)")
                return False, f"CPU使用率({cpu_percent:.1f}%)超过阈值({self.threshold.max_cpu_percent}%)"
            
            # 检查内存使用率
            memory = psutil.virtual_memory()
            logger.info(f"当前内存使用率: {memory.percent:.1f}%")
            if memory.percent > self.threshold.max_memory_percent:
                logger.warning(f"内存使用率({memory.percent:.1f}%)超过阈值({self.threshold.max_memory_percent}%)")
                return False, f"内存使用率({memory.percent:.1f}%)超过阈值({self.threshold.max_memory_percent}%)"
            
            # 检查GPU使用率（如果有）
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for gpu in gpus:
                        gpu_load = gpu.load * 100
                        logger.info(f"GPU {gpu.id}使用率: {gpu_load:.1f}%")
                        if gpu_load > self.threshold.max_gpu_percent:
                            logger.warning(f"GPU {gpu.id}使用率({gpu_load:.1f}%)超过阈值({self.threshold.max_gpu_percent}%)")
                            return False, f"GPU {gpu.id}使用率({gpu_load:.1f}%)超过阈值({self.threshold.max_gpu_percent}%)"
            except Exception as e:
                logger.info(f"GPU检查跳过: {str(e)}")
            
            logger.info("系统资源检查通过")
            return True, "资源充足"

        # 在线程池中执行同步检查
        return await asyncio.get_event_loop().run_in_executor(None, check_sync)

    async def check_resources(self) -> tuple[bool, str]:
        """
        异步检查系统资源是否足够
        :return: (是否有足够资源, 原因描述)
        """
        async with self._lock:
            return await self._check_resources_internal()
    
    async def acquire_resource(self) -> bool:
        """异步获取资源"""
        async with self._lock:
            # 直接调用内部检查方法，避免重复加锁
            resources_available, reason = await self._check_resources_internal()
            if resources_available:
                self.current_tasks += 1
                logger.info(f"成功获取资源，当前任务数: {self.current_tasks}")
                return True
            logger.warning(f"获取资源失败: {reason}")
            return False
    
    async def release_resource(self):
        """异步释放资源"""
        async with self._lock:
            if self.current_tasks > 0:
                self.current_tasks -= 1
                logger.info(f"释放资源完成，当前任务数: {self.current_tasks}")