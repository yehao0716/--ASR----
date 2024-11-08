import pytest
from fastapi.testclient import TestClient
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_server import app

@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)

def test_system_status(client):
    """测试系统状态接口"""
    response = client.get("/system-status")
    assert response.status_code == 200
    data = response.json()
    assert "resources_available" in data
    assert "current_tasks" in data
    assert "timestamp" in data

@pytest.mark.asyncio
async def test_video_to_text(client):
    """测试视频转文字接口"""
    response = client.post(
        "/video-to-text",
        json={
            "video_url": "http://example.com/test.mp4"
        }
    )
    assert response.status_code in [200, 503]  # 可能成功或资源不足
    data = response.json()
    assert "request_id" in data
    assert "process_time" in data
    assert "success" in data 