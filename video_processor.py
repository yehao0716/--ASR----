import warnings
import os

# 禁用特定警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*numpy.core.*")
warnings.filterwarnings("ignore", message=".*aifc is deprecated.*")
warnings.filterwarnings("ignore", message=".*sunau is deprecated.*")

# 禁用 huggingface_hub 的符号链接警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import os
import requests
from moviepy.editor import VideoFileClip
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForTokenClassification
from pydub import AudioSegment
import time
from utils.logger import setup_logger
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import asyncio
from config import Config
from video_downloader import download_video
import logging
import librosa
import jieba
import re
import shutil
from pathlib import Path
from typing import Union
from punctuation_processor import TextProcessor

# 使用已经存在的logger
logger = logging.getLogger('VideoASR')

class VideoSubtitleExtractor:
    def __init__(self, model=None, proxy=None):
        """
        初始化时可以传入预加载的模型
        :param model: 预加载的whisper模型或mock模型
        :param proxy: 代理设置
        """
        if model is None:
            self._init_model(proxy)
        else:
            self.pipe = model
            
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.temp_files = []  # 用于跟踪临时文件
        self.text_processor = TextProcessor(
            mode=Config.TEXT_PROCESS_MODE,  # 在config中配置处理模式
            proxy=proxy
        )
    
    def _init_model(self, proxy=None):
        """同步初始化模型"""
        self.device = Config.DEVICE
        model_path = Config.MODEL_DIR / "whisper-tiny"
        
        try:
            # 如果本地没有模型，则下载
            if not model_path.exists():
                logger.info("正在下载模型到本地...")
                # 使用代理下载模型
                proxies = {'http': proxy, 'https': proxy} if proxy else None
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    Config.MODEL_NAME,
                    torch_dtype=torch.float32,
                    proxies=proxies
                )
                self.processor = AutoProcessor.from_pretrained(
                    Config.MODEL_NAME,
                    proxies=proxies
                )
                # 保存型到本地
                self.model.save_pretrained(model_path)
                self.processor.save_pretrained(model_path)
                logger.info("模型下载完成")
            else:
                logger.info("使用本地模型...")
                # 直接加载本地模型
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32
                )
                self.processor = AutoProcessor.from_pretrained(model_path)
                
            self.model = self.model.to(self.device)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

    # ... [其他方法保持不变] ...

    async def process_video_async(self, video_path):
        """异步处理视频"""
        temp_audio = Config.TEMP_DIR / "temp_audio.wav"
        logger.info(f"开始处理视频: {video_path}")
        try:
            logger.info("开始提取音频...")
            if await self.extract_audio_async(video_path, temp_audio):
                logger.info("音频提取完成，开始分割音频...")
                segments = await self.split_audio_async(temp_audio)
                logger.info(f"音频分割完成，得到 {len(segments)} 个片段，开始并发处理...")
                results = await self.process_segments_async(segments)
                
                raw_text = "\n".join(results)
                logger.info("开始文本后处理...")
                processed_text = self.text_processor.process_text(raw_text)
                logger.info("文本后处理完成")
                
                return processed_text
            return None
        except Exception as e:
            logger.error(f"处理视频时发生错误: {str(e)}")
            raise
        finally:
            logger.info("开始清理临时文件...")
            await self.cleanup_async()
            logger.info("临时文件清理完成")

    async def split_audio_async(self, audio_path, segment_length=Config.SEGMENT_LENGTH):
        """异步分割音频为片段"""
        logger.info(f"开始分割音频文件: {audio_path}")
        def _split():
            try:
                audio = AudioSegment.from_wav(audio_path)
                length_ms = len(audio)
                segment_length_ms = segment_length * 1000
                segments = []
                
                total_segments = (length_ms + segment_length_ms - 1) // segment_length_ms
                logger.info(f"音频总长度: {length_ms/1000:.1f}秒, 将分割为 {total_segments} 个片段")
                
                for i in range(0, length_ms, segment_length_ms):
                    segment = audio[i:i + segment_length_ms]
                    segment_path = Config.TEMP_DIR / f"temp_segment_{i//1000}.wav"
                    segment.export(str(segment_path), format="wav")
                    segments.append(segment_path)
                    logger.debug(f"已导出片段 {len(segments)}/{total_segments}")
                
                logger.info(f"音频分割完成，共生成 {len(segments)} 个片段")
                return segments
            except Exception as e:
                logger.error(f"分割音频时发生错误: {str(e)}")
                raise
        
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            _split
        )

    def extract_audio(self, video_path, output_path):
        """从视频中提取音频"""
        try:
            # 将 Path 对象转换为字符串
            video_path_str = str(video_path)
            output_path_str = str(output_path)
            
            video = VideoFileClip(video_path_str)
            if video.audio is None:
                logger.error("视频文件没有音轨")
                return False
                
            try:
                video.audio.write_audiofile(output_path_str)
                return True
            finally:
                video.close()  # 确保在任何情况下都关闭视频文件
        except Exception as e:
            logger.error(f"音频提取失败: {str(e)}")
            return False

    def transcribe_audio(self, audio_path):
        """同步转换音频为文字"""
        try:
            logger.debug(f"加载音频文件: {audio_path}")
            # 使用 librosa 加载音频文件为 numpy 数组
            audio_array, _ = librosa.load(str(audio_path), sr=16000)
            
            logger.debug(f"开始识别音频: {audio_path}")
            result = self.pipe(
                audio_array,
                return_timestamps=True,
                chunk_length_s=30,
                stride_length_s=5
            )
            logger.debug(f"音频识别完成: {audio_path}")
            return result["text"]
        except Exception as e:
            logger.error(f"语识别败: {str(e)}")
            return None

    async def download_video_async(self, url, local_path):
        """异步下载视频"""
        return await download_video(url, local_path)

    async def extract_audio_async(self, video_path: Union[str, Path], audio_path: Union[str, Path]) -> bool:
        """异步提取音频"""
        try:
            # 确保输出目录存在
            output_dir = Path(audio_path).parent
            if not output_dir.exists():
                logger.error(f"输出目录不存在: {output_dir}")
                return False
                
            video_path_str = str(video_path)
            audio_path_str = str(audio_path)
            
            video = VideoFileClip(video_path_str)
            if video.audio is None:
                logger.error("视频文件没有音轨")
                return False
                
            try:
                video.audio.write_audiofile(audio_path_str)
                return True
            finally:
                video.close()
        except Exception as e:
            logger.error(f"音频提取失败: {str(e)}")
            # 清理临时文件
            temp_path = Path(f"{audio_path_str}.temp")
            if temp_path.exists():
                temp_path.unlink()
            if Path(audio_path_str).exists():
                Path(audio_path_str).unlink()
            return False

    async def transcribe_audio_async(self, audio_path):
        """步转换音频为文字"""
        logger.debug(f"开始转写音频: {audio_path}")
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.transcribe_audio,
                audio_path
            )
            if result:
                logger.debug(f"音频转写完成: {audio_path}, 文本长度: {len(result)}")
            else:
                logger.warning(f"音频转写失败: {audio_path}")
            return result
        except Exception as e:
            logger.error(f"转写音频时发生错误 {audio_path}: {str(e)}")
            return None

    async def cleanup_async(self):
        """异步清理临时文件"""
        def cleanup():
            for file in self.temp_files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except Exception as e:
                    logger.error(f"清理文件失败 {file}: {str(e)}")
            self.temp_files.clear()
        
        await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            cleanup
        )

    async def process_segments_async(self, segments):
        """并发处理所有音频片段"""
        logger.info(f"开始并发处理 {len(segments)} 个音频片段")
        tasks = []
        for i, segment in enumerate(segments):
            logger.debug(f"创建任务 {i+1}/{len(segments)}: {segment}")
            task = self.transcribe_audio_async(segment)
            tasks.append(task)
        
        try:
            logger.info("等待所有转写任务完成...")
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r]
            logger.info(f"音频处理完成，成功率: {len(valid_results)}/{len(segments)}")
            return valid_results
        except Exception as e:
            logger.error(f"处理音频片段时发生错误: {str(e)}")
            raise

async def process_video_url_async(video_url, request_id):
    """
    异步处理视频URL
    :param video_url: 视频URL
    :param request_id: 请求ID
    :return: 处理结果
    """
    try:
        temp_video = Config.TEMP_DIR / "temp_video.mp4"
        logger.info(f"[{request_id}] 初始化视频处理器...")
        try:
            extractor = VideoSubtitleExtractor(proxy=Config.PROXY)
            logger.info(f"[{request_id}] 视频处理器初始化完成")
        except Exception as e:
            logger.error(f"[{request_id}] 视频处理器初始化失败: {str(e)}")
            raise
        
        # 异步下载视频
        logger.info(f"[{request_id}] 开始调用下载函数...")
        try:
            download_result = await download_video(video_url, temp_video, request_id)
            logger.info(f"[{request_id}] 下载函数返回结果: {download_result}")
            if not download_result:
                raise Exception("视频下载失败")
        except Exception as e:
            logger.error(f"[{request_id}] 下载过程出错: {str(e)}")
            raise
            
        logger.info(f"[{request_id}] 视频下载完成，准备处理音频...")
        
        # 异步处理视频
        logger.info(f"[{request_id}] 开始提取音频...")
        text = await extractor.process_video_async(temp_video)
        if not text:
            raise Exception("音频处理失败")
        
        logger.info(f"[{request_id}] 提取文字长度: {len(text)} 字符")
        
        return {
            "success": True,
            "text": text
        }
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }
    finally:
        # 确保清理临时文件
        try:
            await extractor.cleanup_async()
            logger.info(f"[{request_id}] 临时文件清理完成")
        except Exception as e:
            logger.error(f"[{request_id}] 清理临时文件失败: {str(e)}")