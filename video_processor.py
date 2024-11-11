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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
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
        self.text_processor = TextPostProcessor(proxy=proxy)
    
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

class TextPostProcessor:
    def __init__(self, proxy=None):
        """初始化后处理器"""
        # 初始化错别字修正字
        self.corrections = {
            # 常见错误（优先级最高）
            '特别值的': '特别值得',  # 添加完整短语
            '值的': '值得',         # 基本词组
            
            # 基本错别字
            '呐': '那',
            '阿郎': '阿朗',
            '漏面': '露面',
            '孑': '子',
            
            # 繁简转换（按字处理）
            '這': '这',
            '裡': '里',
            '這裡': '这里',  # 组合词要放在单字后面
            '們': '们',
            '妳': '你',
            
            # 其他繁体字
            '產': '产',
            '為': '为',
            '會': '会',
            '機': '机',
            '關': '关',
            '開': '开',
            '時': '时',
            '長': '长',
            '邊': '边',
            '後': '后',
            '發': '发',
            '說': '说',
            '當': '当',
        }
        
        logger.info("开始初始化文本后处理器...")
        model_path = Config.MODEL_DIR / "text2text-correction-chinese"
        offload_folder = Config.TEMP_DIR / "model_offload"
        
        try:
            if model_path.exists():
                logger.info("使用本地模型...")
                self.punctuation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    device_map='auto',
                    offload_folder=str(offload_folder)
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    local_files_only=True
                )
            else:
                logger.info("本地模型不存在，从 HuggingFace 下载...")
                proxies = {'http': proxy, 'https': proxy} if proxy else None
                
                self.punctuation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    "fnlp/bart-base-chinese",
                    proxies=proxies,
                    trust_remote_code=True,
                    device_map='auto',
                    offload_folder=str(offload_folder)
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "fnlp/bart-base-chinese",
                    proxies=proxies,
                    trust_remote_code=True
                )
                
                logger.info(f"保存模型到本地: {model_path}")
                model_path.mkdir(parents=True, exist_ok=True)
                self.punctuation_model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
            
            self._verify_model()
            logger.info("模型加载成功")
            self.model_available = True
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            self.model_available = False
        finally:
            if offload_folder.exists():
                shutil.rmtree(offload_folder)

    def _verify_model(self):
        """验证模型是否可用"""
        logger.info("开始验证模型...")
        test_text = "测试文本"
        try:
            inputs = self.tokenizer(
                test_text, 
                return_tensors="pt", 
                truncation=True,
                return_token_type_ids=False
            )
            inputs = {k: v.to(self.punctuation_model.device) for k, v in inputs.items()}
            outputs = self.punctuation_model.generate(**inputs)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"模型验证成功: '{test_text}' -> '{result}'")
        except Exception as e:
            logger.error(f"模型验证失败: {str(e)}")
            raise

    def _process_with_model(self, text):
        """使用模型处理文本"""
        if not self.model_available or not text:
            logger.info("模型不可用或文本为空，跳过模型处理")
            return text
            
        try:
            logger.info(f"使用模型处理文本: '{text}'")
            
            # 保护英文单词和数字
            protected_words = {}
            def protect_word(match):
                word = match.group(0)
                token = f"__PROTECTED_{len(protected_words)}__"
                protected_words[token] = word
                return token
                
            # 保护英文单词（包括前后的空格）
            text = re.sub(r'(\s*[a-zA-Z]+\s*)', protect_word, text)
            
            # 2. 优化生成配置
            generation_config = GenerationConfig(
                max_new_tokens=min(len(text) * 2, 512),
                num_beams=8,  # 增加搜索宽度
                length_penalty=1.0,  # 降低长句子的惩罚
                early_stopping=True,
                do_sample=False,
                temperature=0.8,  # 降低温度，使输出更保守
                no_repeat_ngram_size=4,  # 增加不重复片段长度
                repetition_penalty=1.5,  # 增加重复惩罚
                num_return_sequences=1,
                bad_words_ids=None,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                diversity_penalty=0.0,
                top_k=50,
                top_p=0.9,
            )
            
            # 3. 优化模型输入
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
                max_length=256,
                return_token_type_ids=False,
                padding=True,
                add_special_tokens=True
            )
            inputs = {k: v.to(self.punctuation_model.device) for k, v in inputs.items()}
            
            # 4. 模型生成
            with torch.no_grad():
                outputs = self.punctuation_model.generate(
                    **inputs,
                    generation_config=generation_config
                )
                
            # 5. 解码并后处理
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"模型处理结果: '{result}'")
            
            # 恢复保护的单词
            for token, word in protected_words.items():
                result = result.replace(token, word)
                
            return result
            
        except Exception as e:
            logger.error(f"模型处理失败: {str(e)}")
            return text

    def _apply_rules(self, text):
        """应用规则处理文本"""
        if not text:
            return text
            
        try:
            # 保护英文单词和数字
            protected_tokens = {}
            def protect_token(match):
                word = match.group(0)
                token = f"__TOKEN_{len(protected_tokens)}__"
                protected_tokens[token] = word
                return token
                
            # 保护英文单词（包括前后的空格）
            text = re.sub(r'(\s*[a-zA-Z]+\s*)', protect_token, text)
            
            # 2. 错别字修正
            sorted_corrections = sorted(
                self.corrections.items(),
                key=lambda x: len(x[0]),
                reverse=True
            )
            for wrong, right in sorted_corrections:
                text = text.replace(wrong, right)
            
            # 3. 清理文本和标点
            text = re.sub(r'\s+', '', text)
            text = re.sub(r'[#￥%&*@……\[\]\{\}\<\>~]+', '', text)
            
            # 4. 标点处理规则（按优先级排序）
            patterns = [
                # 1. 处理感叹和问句（最高优先级）
                (r'([啊呀哇])([^。！？]|$)', r'\1！'),
                (r'([吗么呢])([^，。！？]|$)', r'\1？'),
                
                # 2. 处理连接词（第二优先级）
                (r'([^，。！？])但是还是([^，。！？])', r'\1，但是还是\2'),
                (r'([。！？])但是还是', r'\1 但是还是'),
                
                # 3. 处理句子分割（第三优先级）
                (r'([^，。！？])今天([^，。！？])', r'\1。今天\2'),
                (r'([^，。！？])要上班([^，。！？])', r'\1，要上班\2'),
                
                # 4. 处理特殊符号
                (r'[:：]([^，。！？])', r'：\1'),
                
                # 5. 处理数字
                (r'(\d{1,3})(?=(\d{3})+(?!\d))', r'\1,'),
                (r'(\d+)([元块角分])', r'\1\2，'),
            ]
            
            for pattern, repl in patterns:
                text = re.sub(pattern, repl, text)
            
            # 5. 恢复保护的标记
            for token, original in protected_tokens.items():
                text = text.replace(token, original)
            
            # 确保英文单词前后有正确的空格和标点
            text = re.sub(r'([a-zA-Z])([，。！？])', r'\1 \2', text)
            text = re.sub(r'([，。！？])([a-zA-Z])', r'\1 \2', text)
            
            return text
            
        except Exception as e:
            logger.error(f"规则处理失败: {str(e)}")
            return text or ""

    def process_text(self, text):
        """处理文本：结合模型和规则"""
        if text is None:
            return None
        if not text:
            return ""
            
        try:
            logger.info("开始处理文本...")
            
            # 处理超长文本
            if len(text) > 5000:
                logger.info(f"检测到超长文本，长度={len(text)}，进行分段处理")
                # 分批处理超长文本
                chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                processed_chunks = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"处理第 {i+1}/{len(chunks)} 段")
                    result = self._process_chunk(chunk)
                    if result:
                        processed_chunks.append(result)
                
                # 确保返回非空结果
                if processed_chunks:
                    final_result = "".join(processed_chunks)
                    logger.info(f"超长文本处理完成，长度={len(final_result)}")
                    return final_result
                
                # 如果分段处理失败，使用规则处理原文本
                logger.warning("分段处理失败，使用规则处理原文本")
                return self._apply_rules(text)
            
            # 处理普通长度文本
            result = self._process_chunk(text)
            if not result:
                logger.warning("普通文本处理失败，使用规则处理")
                result = self._apply_rules(text)
                
            return result
            
        except Exception as e:
            logger.error(f"文本处理失败: {str(e)}")
            return self._apply_rules(text)

    def _process_chunk(self, text):
        """处理单个文本块"""
        try:
            # 1. 先用模型处理
            if self.model_available:
                result = self._process_with_model(text)
            else:
                result = text
                
            # 2. 再应用规则
            result = self._apply_rules(result)
            
            # 3. 确保返回非空结果
            if not result:
                logger.warning("文本块处理结果为空，使用规则处理原文本")
                result = self._apply_rules(text)
                
            return result
            
        except Exception as e:
            logger.error(f"文本块处理失败: {str(e)}")
            return self._apply_rules(text)

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