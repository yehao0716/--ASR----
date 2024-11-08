import pytest
import os
from pathlib import Path
import subprocess
from video_processor import VideoSubtitleExtractor

@pytest.fixture(scope="module")
def sample_video_no_audio(tmp_path_factory):
    """创建一个无音轨的测试视频"""
    tmp_path = tmp_path_factory.mktemp("video_test")
    video_path = tmp_path / "test_video_no_audio.mp4"
    
    cmd = [
        'ffmpeg', '-f', 'lavfi',
        '-i', 'color=c=black:s=640x480:d=1',
        '-c:v', 'libx264',
        str(video_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path

@pytest.fixture(scope="module")
def invalid_video(tmp_path_factory):
    """创建一个无效的视频文件"""
    tmp_path = tmp_path_factory.mktemp("video_test")
    video_path = tmp_path / "invalid.mp4"
    
    # 创建一个无效的视频文件
    with open(video_path, 'wb') as f:
        f.write(b'invalid data')
    
    return video_path

@pytest.mark.asyncio
async def test_nonexistent_file(video_extractor, tmp_path):
    """测试处理不存在的文件"""
    nonexistent_path = tmp_path / "nonexistent.mp4"
    audio_path = tmp_path / "audio.wav"
    
    result = await video_extractor.extract_audio_async(str(nonexistent_path), str(audio_path))
    assert result == False
    assert not audio_path.exists()

@pytest.mark.asyncio
async def test_invalid_video(video_extractor, invalid_video, tmp_path):
    """测试处理无效的视频文件"""
    audio_path = tmp_path / "audio.wav"
    result = await video_extractor.extract_audio_async(str(invalid_video), str(audio_path))
    assert result == False
    assert not audio_path.exists()

@pytest.mark.asyncio
async def test_no_audio_video(video_extractor, sample_video_no_audio, tmp_path):
    """测试处理无音轨视频"""
    audio_path = tmp_path / "audio.wav"
    result = await video_extractor.extract_audio_async(str(sample_video_no_audio), str(audio_path))
    assert result == False
    assert not audio_path.exists()

@pytest.mark.asyncio
async def test_permission_error(video_extractor, sample_video, tmp_path):
    """测试权限错误"""
    if os.name != 'nt':  # 在非Windows系统上测试权限
        audio_path = tmp_path / "audio.wav"
        os.chmod(tmp_path, 0o444)  # 设置只读权限
        
        try:
            result = await video_extractor.extract_audio_async(str(sample_video), str(audio_path))
            assert result == False
            assert not audio_path.exists()
        finally:
            os.chmod(tmp_path, 0o777)  # 恢复权限

@pytest.mark.asyncio
async def test_cleanup_nonexistent_file(video_extractor):
    """测试清理不存在的文件"""
    # 添加一个不存在的文件到临时文件列表
    video_extractor.temp_files.append("nonexistent.wav")
    
    # 清理应该不会抛出异常
    await video_extractor.cleanup_async()

@pytest.mark.asyncio
async def test_extract_audio_with_output_in_nonexistent_directory(video_extractor, sample_video, tmp_path):
    """测试输出到不存在的目录"""
    nonexistent_dir = tmp_path / "nonexistent_dir" / "audio.wav"
    result = await video_extractor.extract_audio_async(str(sample_video), str(nonexistent_dir))
    assert result == False
    assert not nonexistent_dir.exists()