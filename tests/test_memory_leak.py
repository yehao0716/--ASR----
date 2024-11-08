import pytest
import psutil
import gc
import time
import subprocess
from pathlib import Path
from video_processor import VideoSubtitleExtractor

def get_memory_usage():
    """获取当前进程的内存使用"""
    return psutil.Process().memory_info().rss / 1024 / 1024  # MB

@pytest.fixture(scope="module")
def mock_whisper_model():
    """提供一个mock的whisper模型"""
    class MockWhisperPipeline:
        def __call__(self, audio_path, **kwargs):
            return {"text": "这是一段测试文本"}
    return MockWhisperPipeline()

@pytest.fixture(scope="module")
def video_extractor(mock_whisper_model):
    """创建视频处理器实例"""
    return VideoSubtitleExtractor(model=mock_whisper_model)

@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """创建一个标准测试视频"""
    tmp_path = tmp_path_factory.mktemp("video_test")
    video_path = tmp_path / "test_video.mp4"
    
    cmd = [
        'ffmpeg', '-f', 'lavfi',
        '-i', 'sine=frequency=1000:duration=1',
        '-f', 'lavfi',
        '-i', 'color=c=black:s=640x480:d=1',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        str(video_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path

@pytest.mark.asyncio
async def test_memory_leak(video_extractor, sample_video, tmp_path):
    """测试内存泄漏"""
    initial_memory = get_memory_usage()
    
    # 执行多次处理
    for i in range(5):
        audio_path = tmp_path / f"audio_{i}.wav"
        await video_extractor.extract_audio_async(str(sample_video), str(audio_path))
        await video_extractor.cleanup_async()
        
        # 强制垃圾回收
        gc.collect()
        time.sleep(1)  # 等待系统回收资源
        
    final_memory = get_memory_usage()
    
    # 验证内存使用是否稳定
    assert abs(final_memory - initial_memory) < 50  # 允许50MB的波动