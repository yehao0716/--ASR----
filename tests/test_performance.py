import pytest
import os
import asyncio
import psutil
import time
from pathlib import Path
import subprocess
from video_processor import VideoSubtitleExtractor, process_video_url_async

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
    start_time = time.time()
    print("开始初始化视频处理器...")
    extractor = VideoSubtitleExtractor(model=mock_whisper_model)
    print(f"视频处理器初始化完成，耗时: {time.time() - start_time:.2f}秒")
    return extractor

@pytest.fixture(scope="module")
def large_video(tmp_path_factory):
    """创建一个较大的测试视频（10秒）"""
    tmp_path = tmp_path_factory.mktemp("video_test")
    video_path = tmp_path / "large_video.mp4"
    
    cmd = [
        'ffmpeg', '-f', 'lavfi',
        '-i', 'sine=frequency=1000:duration=10',
        '-f', 'lavfi',
        '-i', 'color=c=black:s=1920x1080:d=10',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        str(video_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path

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
async def test_large_file_processing(video_extractor, large_video, tmp_path):
    """测试处理大文件"""
    start_time = time.time()
    audio_path = tmp_path / "large_audio.wav"
    
    # 记录初始内存使用
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    result = await video_extractor.extract_audio_async(str(large_video), str(audio_path))
    assert result == True
    
    # 记录处理后内存使用
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    processing_time = time.time() - start_time
    
    # 验证性能指标
    assert processing_time < 30  # 处理时间应该在30秒内
    assert final_memory - initial_memory < 500  # 内存增长不应超过500MB

@pytest.mark.asyncio
async def test_concurrent_processing(video_extractor, sample_video, tmp_path):
    """测试并发处理"""
    num_concurrent = 3
    audio_paths = [tmp_path / f"audio_{i}.wav" for i in range(num_concurrent)]
    
    # 创建多个并发任务
    tasks = [
        video_extractor.extract_audio_async(str(sample_video), str(audio_path))
        for audio_path in audio_paths
    ]
    
    # 并发执行
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    processing_time = time.time() - start_time
    
    # 验证结果
    assert all(results)  # 所有任务都应该成功
    assert processing_time < 10  # 总处理时间应该小于单个任务的3倍