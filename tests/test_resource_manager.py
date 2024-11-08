import pytest
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.resource_manager import ResourceManager, ResourceThreshold

@pytest.fixture
def resource_manager():
    """创建资源管理器实例"""
    return ResourceManager(
        ResourceThreshold(
            max_cpu_percent=80.0,
            max_memory_percent=80.0,
            max_gpu_percent=80.0,
            max_concurrent_tasks=2
        )
    )

@pytest.mark.asyncio
async def test_acquire_release_resource(resource_manager):
    """测试资源获取和释放"""
    # 获取资源
    result = await resource_manager.acquire_resource()
    assert result == True
    assert resource_manager.current_tasks == 1
    
    # 再次获取资源
    result = await resource_manager.acquire_resource()
    assert result == True
    assert resource_manager.current_tasks == 2
    
    # 第三次获取应该失败（超过最大任务数）
    result = await resource_manager.acquire_resource()
    assert result == False
    assert resource_manager.current_tasks == 2
    
    # 释放资源
    await resource_manager.release_resource()
    assert resource_manager.current_tasks == 1
    
    # 再次释放
    await resource_manager.release_resource()
    assert resource_manager.current_tasks == 0 