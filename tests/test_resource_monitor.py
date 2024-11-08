import pytest
import sys
import os
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.resource_monitor import ResourceMonitor

@pytest.fixture
def resource_monitor():
    """创建资源监控器实例"""
    monitor = ResourceMonitor(log_interval=1)
    yield monitor
    monitor.stop()  # 确保测试后停止监控

def test_resource_monitor_start_stop(resource_monitor):
    """测试资源监控器的启动和停止"""
    assert not resource_monitor.is_running
    resource_monitor.start()
    assert resource_monitor.is_running
    resource_monitor.stop()
    assert not resource_monitor.is_running

@pytest.mark.asyncio
async def test_resource_monitor_logging(resource_monitor):
    """测试资源监控器的日志记录"""
    resource_monitor.start()
    await asyncio.sleep(2)  # 等待足够时间生成日志
    resource_monitor.stop()
    # 验证是否生成了日志
    # 这里需要检查日志文件或捕获日志输出 