import pytest
from pathlib import Path
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config

def test_config_setup():
    """测试配置初始化"""
    Config.setup()
    assert Config.TEMP_DIR.exists()
    assert Config.LOG_DIR.exists()

def test_config_values():
    """测试配置值"""
    assert isinstance(Config.CPU_THRESHOLD, float)
    assert isinstance(Config.MEMORY_THRESHOLD, float)
    assert isinstance(Config.GPU_THRESHOLD, float)
    assert isinstance(Config.MAX_CONCURRENT_TASKS, int)
    assert isinstance(Config.SEGMENT_LENGTH, int) 