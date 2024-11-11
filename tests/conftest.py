import pytest
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import logging
import time
import subprocess
from pathlib import Path
from video_processor import VideoSubtitleExtractor
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config  # 添加这行导入

logger = logging.getLogger('VideoASR')

@pytest.fixture(scope="session")
def whisper_model():
    """
    会话级别的fixture，加载一个较小的whisper模型
    所有测试共享同一个模型实例
    """
    start_time = time.time()
    logger.info("开始加载Whisper模型...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 使用tiny模型代替base模型，大小只有base模型的1/5
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-tiny",  # 改用tiny模型
        torch_dtype=torch.float32
    ).to(device)
    
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )
    
    logger.info(f"Whisper模型加载完成，耗时: {time.time() - start_time:.2f}秒")
    return pipe

@pytest.fixture(scope="session")
def mock_whisper_model():
    """提供一个mock的whisper模型"""
    class MockWhisperPipeline:
        def __init__(self):
            self.model_path = Config.MODEL_DIR / "whisper-tiny"
            
        def __call__(self, audio_path, **kwargs):
            return {"text": "这是一段测试文本"}
            
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            
        def to(self, device):
            return self
            
    return MockWhisperPipeline()

@pytest.fixture(scope="session")
def video_extractor(mock_whisper_model):
    """创建视频处理器实例"""
    return VideoSubtitleExtractor(model=mock_whisper_model)

@pytest.fixture(scope="session")
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