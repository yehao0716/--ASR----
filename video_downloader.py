import aiohttp
import uuid
import logging
import asyncio
import aiofiles
import time
import os
from typing import List, Tuple
from config import Config

# 使用已经存在的logger
logger = logging.getLogger('VideoASR')

class DownloadChunk:
    def __init__(self, start: int, end: int, data: bytes = None):
        self.start = start
        self.end = end
        self.data = data
        self.size = end - start + 1

async def download_chunk(session: aiohttp.ClientSession, url: str, chunk: DownloadChunk, request_id: str) -> bool:
    """下载指定范围的数据块"""
    headers = {'Range': f'bytes={chunk.start}-{chunk.end}'}
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 206:  # 部分内容
                chunk.data = await response.read()
                return True
            else:
                logger.error(f"[{request_id}] 下载分片失败，状态码: {response.status}")
                return False
    except Exception as e:
        logger.error(f"[{request_id}] 下载分片出错: {str(e)}")
        return False

async def download_video(url: str, save_path: str, request_id: str) -> bool:
    temp_path = f"{save_path}.temp"
    try:
        logger.info(f"[{request_id}] 开始下载视频: {url}")
        
        timeout = aiohttp.ClientTimeout(
            total=Config.DOWNLOAD_TIMEOUT,
            connect=Config.CONNECT_TIMEOUT,
            sock_read=Config.READ_TIMEOUT
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                # 获取文件大小
                async with session.head(url) as response:
                    if response.status != 200:
                        logger.error(f"[{request_id}] 无法获取文件信息，状态码: {response.status}")
                        return False
                        
                    total_size = int(response.headers.get('content-length', 0))
                    if total_size == 0:
                        logger.error(f"[{request_id}] 无法获取文件大小")
                        return False
                    
                    logger.info(f"[{request_id}] 文件大小: {total_size / (1024*1024):.2f} MB")
                    
                    # 创建下载分片
                    chunks: List[DownloadChunk] = []
                    chunk_count = (total_size + Config.DOWNLOAD_CHUNK_SIZE - 1) // Config.DOWNLOAD_CHUNK_SIZE
                    for i in range(chunk_count):
                        start = i * Config.DOWNLOAD_CHUNK_SIZE
                        end = min(start + Config.DOWNLOAD_CHUNK_SIZE - 1, total_size - 1)
                        chunks.append(DownloadChunk(start, end))
                    
                    logger.info(f"[{request_id}] 将使用 {len(chunks)} 个分片下载，每个分片大小: {Config.DOWNLOAD_CHUNK_SIZE / (1024*1024):.1f}MB")
                    
                    # 并发下载分片
                    start_time = time.time()
                    last_progress_time = start_time
                    downloaded_bytes = 0
                    
                    try:
                        async with aiofiles.open(temp_path, 'wb') as f:
                            for i in range(0, len(chunks), Config.MAX_CONCURRENT_CHUNKS):
                                batch = chunks[i:i + Config.MAX_CONCURRENT_CHUNKS]
                                batch_start = time.time()
                                logger.info(f"[{request_id}] 开始下载分片 {i+1}-{i+len(batch)} / {len(chunks)}")
                                
                                tasks = [download_chunk(session, url, chunk, request_id) for chunk in batch]
                                results = await asyncio.gather(*tasks)
                                
                                if not all(results):
                                    logger.error(f"[{request_id}] 部分分片下载失败")
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                    return False
                                
                                # 按顺序写入文件
                                for chunk in batch:
                                    if not chunk.data:
                                        logger.error(f"[{request_id}] 分片数据为空")
                                        if os.path.exists(temp_path):
                                            os.remove(temp_path)
                                        return False
                                        
                                    await f.write(chunk.data)
                                    downloaded_bytes += len(chunk.data)
                                    chunk.data = None  # 释放内存
                                    
                                    # 每次写入后都更新进度，但限制日志频率
                                    current_time = time.time()
                                    if current_time - last_progress_time >= 1.0:  # 确保至少1秒间隔
                                        progress = (downloaded_bytes / total_size) * 100
                                        speed = downloaded_bytes / (1024*1024) / (current_time - start_time)  # MB/s
                                        remaining_bytes = total_size - downloaded_bytes
                                        eta = remaining_bytes / (speed * 1024*1024) if speed > 0 else 0
                                        
                                        logger.info(
                                            f"[{request_id}] 下载进度: {progress:.1f}% "
                                            f"({downloaded_bytes/(1024*1024):.1f}MB/{total_size/(1024*1024):.1f}MB), "
                                            f"速度: {speed:.2f}MB/s, "
                                            f"预计剩余时间: {eta/60:.1f}分钟"
                                        )
                                        last_progress_time = current_time
                                
                                batch_time = time.time() - batch_start
                                logger.info(f"[{request_id}] 分片 {i+1}-{i+len(batch)} 下载完成，耗时: {batch_time:.1f}秒")
                        
                        # 验证文件大小
                        actual_size = os.path.getsize(temp_path)
                        if actual_size != total_size:
                            logger.error(f"[{request_id}] 文件大小不匹配，期望: {total_size}，实际: {actual_size}")
                            os.remove(temp_path)
                            return False
                        
                        # 重命名临时文件
                        if os.path.exists(save_path):
                            os.remove(save_path)
                        os.rename(temp_path, save_path)
                        
                        total_time = time.time() - start_time
                        average_speed = (total_size / (1024*1024)) / total_time
                        logger.info(
                            f"[{request_id}] 下载完成，总耗时: {total_time:.1f}秒, "
                            f"平均速度: {average_speed:.2f}MB/s"
                        )
                        return True
                        
                    except Exception as e:
                        logger.error(f"[{request_id}] 下载过程发生错误: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        return False
                        
            except aiohttp.ClientError as e:
                logger.error(f"[{request_id}] 网络错误: {str(e)}")
                return False
                    
    except Exception as e:
        logger.error(f"[{request_id}] 下载过程发生错误: {str(e)}")
        return False
    finally:
        # 确保清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"[{request_id}] 清理临时文件失败: {str(e)}")

# 现有代码... 