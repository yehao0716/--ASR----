import pytest
import aiohttp
import os
import sys
from pathlib import Path
import time
import asyncio
import logging

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from video_downloader import download_video
from config import Config

# 在文件开头添加常量
TEST_FILE_SIZE_SMALL = 50 * 1024 * 1024   # 50MB，用于简单测试
TEST_FILE_SIZE_LARGE = 100 * 1024 * 1024  # 100MB，用于进度日志测试
TEST_DELAY = 1.0  # 1秒延迟

class MockResponse:
    """模拟 aiohttp 响应"""
    def __init__(self, status=200, content_length=1024, support_range=True, delay=0.2, error=None):
        self.status = status
        self._total_size = content_length
        self.headers = {
            "content-length": str(content_length),
            "accept-ranges": "bytes" if support_range else "none"
        }
        self._content_pos = 0
        self._data = b"x" * content_length
        self._delay = delay
        self._error = error
        
    async def read(self):
        """读取全部数据"""
        if self._error:
            raise aiohttp.ClientError(f"模拟的网络错误: {self._error}")
        await asyncio.sleep(self._delay)
        # 确保返回正确范围的数据
        if 'Range' in self.headers:
            start, end = map(int, self.headers['Range'].split('=')[1].split('-'))
            return self._data[start:end+1]
        return self._data
        
    @property
    def content(self):
        if self._error:
            raise aiohttp.ClientError(f"模拟的网络错误: {self._error}")
        return self.MockContent(0, self._total_size, self._data, self._delay)
        
    class MockContent:
        def __init__(self, start_pos, total_size, data, delay):
            self.pos = start_pos
            self.total_size = total_size
            self._data = data
            self._delay = delay
            self._chunk_count = 0
            
        async def iter_chunked(self, size):
            # 使用Config中设置的块大小
            chunk_size = Config.DOWNLOAD_CHUNK_SIZE  # 2MB
            while self.pos < self.total_size:
                self._chunk_count += 1
                # 每个块都等待，以确保有足够的时间记录进度
                await asyncio.sleep(self._delay)
                
                end_pos = min(self.pos + chunk_size, self.total_size)
                chunk = self._data[self.pos:end_pos]
                self.pos = end_pos
                if not chunk:
                    break
                yield chunk

class MockGet:
    """模拟 get 请求的响应"""
    def __init__(self, response, headers=None):
        self.response = response
        if headers:
            self.response.headers = headers.copy()  # 保存请求头
        
    async def __aenter__(self):
        return self.response
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockClientSession:
    """模拟 aiohttp ClientSession"""
    def __init__(self, response):
        self.response = response
        self.headers = {}
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def get(self, url, headers=None, **kwargs):
        """返回一个支持异步上下文管理器的对象"""
        self.headers = headers or {}
        if 'Range' in self.headers:
            # 处理断点续传请求
            range_header = self.headers['Range']
            start_bytes = int(range_header.split('=')[1].split('-')[0])
            self.response._content_pos = start_bytes
            self.response.status = 206  # 使用206状态码表示部分内容
            # 修改 content-length 为剩余大小
            remaining_size = self.response._total_size - start_bytes
            self.response.headers["content-length"] = str(remaining_size)
            # 添加 Content-Range 头部
            self.response.headers["content-range"] = f"bytes {start_bytes}-{self.response._total_size-1}/{self.response._total_size}"
        return MockGet(self.response, headers)

    def head(self, url, **kwargs):
        """模拟 head 请求"""
        return MockGet(self.response)

class SizeMismatchResponse(MockResponse):
    """模拟大小不匹配的响应"""
    def __init__(self, status=200, content_length=1024):
        super().__init__(status=status, content_length=content_length)
        # 声明的大小是 content_length，但实际数据只有一半
        self._actual_size = content_length // 2
        self._data = b"x" * self._actual_size
        # 保持 headers 中声明的大小为完整大小
        self.headers = {
            "content-length": str(content_length),
            "accept-ranges": "bytes"
        }
        
    @property
    def content(self):
        # 返回实际大小的 MockContent
        return self.MockContent(0, self._actual_size, self._data, self._delay)

class ErrorResponse(MockResponse):
    """模拟错误响应"""
    def __init__(self, error_type="network"):
        super().__init__(status=200, content_length=1024)
        # 修改这里，直接传递完整的错误消息
        self._error = "模拟的网络错误"  # 不再使用 error_type 参数

@pytest.mark.asyncio
async def test_download_video_success(tmp_path, caplog):
    """测试成功下载视频，并验证日志输出"""
    caplog.set_level(logging.INFO)
    
    url = "http://example.com/test.mp4"
    save_path = tmp_path / "test.mp4"
    request_id = "Test-001"
    
    try:
        # 确保临时文件不存在
        temp_path = Path(f"{save_path}.temp")
        if temp_path.exists():
            temp_path.unlink()
            
        # 确保目标文件不存在
        if save_path.exists():
            save_path.unlink()
        
        mock_response = MockResponse(
            status=200, 
            content_length=TEST_FILE_SIZE_LARGE,  # 100MB
            delay=TEST_DELAY
        )
        mock_session = MockClientSession(mock_response)
        original_session = aiohttp.ClientSession
        aiohttp.ClientSession = lambda *args, **kwargs: mock_session
        
        result = await download_video(url, save_path, request_id)
        
        # 验证下载结果
        assert result == True
        assert save_path.exists()
        assert os.path.getsize(save_path) == TEST_FILE_SIZE_LARGE
        
        # 验证日志输出
        log_text = "\n".join(record.message for record in caplog.records)
        print("\n实际日志输出:")
        print(log_text)
        
        # 检查关键日志是否存在
        assert "[Test-001] 开始下载视频" in log_text
        assert "[Test-001] 文件大小: 100.00 MB" in log_text
        assert "[Test-001] 下载进度" in log_text
        assert "[Test-001] 下载完成" in log_text
        
        # 验证进度日志数量
        progress_logs = [r for r in caplog.records if "下载进度" in r.message]
        assert len(progress_logs) >= 3, "应该至少有3条进度日志"  # 开始、中间、结束
        
    finally:
        aiohttp.ClientSession = original_session
        # 确保清理文件
        for path in [save_path, temp_path]:
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

@pytest.mark.asyncio
async def test_download_resume(tmp_path):
    """测试断点续传功能"""
    url = "http://example.com/test.mp4"
    save_path = tmp_path / "test.mp4"
    temp_path = Path(f"{save_path}.temp")
    request_id = "Test-002"
    
    total_size = 1024*1024  # 总大小1MB
    partial_size = 512*1024  # 已下载512KB
    
    # 创建一个部分下载的临时文件
    with open(temp_path, 'wb') as f:
        f.write(b"x" * partial_size)
    
    try:
        # 创建模拟响应，设置总大小为1MB
        mock_response = MockResponse(status=200, content_length=total_size)
        mock_session = MockClientSession(mock_response)
        original_session = aiohttp.ClientSession
        aiohttp.ClientSession = lambda *args, **kwargs: mock_session
        
        result = await download_video(url, save_path, request_id)
        
        assert result == True
        assert save_path.exists()
        assert not temp_path.exists()  # 临时文件应该被重名
        assert os.path.getsize(save_path) == total_size  # 总大小应该是1MB
        assert 'Range' in mock_session.headers  # 应该使用了Range头部
        
    finally:
        aiohttp.ClientSession = original_session

@pytest.mark.asyncio
async def test_download_no_range_support(tmp_path):
    """测试服务器不支持断点续传的情况"""
    url = "http://example.com/test.mp4"
    save_path = tmp_path / "test.mp4"
    temp_path = Path(f"{save_path}.temp")
    request_id = "Test-003"
    
    total_size = 1024*1024  # 总大小1MB
    partial_size = 512*1024  # 已下载512KB
    
    # 创建一个部分下载的临时文件
    with open(temp_path, 'wb') as f:
        f.write(b"x" * partial_size)
    
    try:
        mock_response = MockResponse(
            status=200, 
            content_length=total_size,  # 完整文件大小1MB
            support_range=False
        )
        mock_session = MockClientSession(mock_response)
        original_session = aiohttp.ClientSession
        aiohttp.ClientSession = lambda *args, **kwargs: mock_session
        
        result = await download_video(url, save_path, request_id)
        
        assert result == True
        assert save_path.exists()
        assert not temp_path.exists()
        assert os.path.getsize(save_path) == total_size  # 总大小应该是1MB
        
    finally:
        aiohttp.ClientSession = original_session

@pytest.mark.asyncio
async def test_download_size_mismatch(tmp_path):
    """测试下载文件大小不匹配的情况"""
    url = "http://example.com/test.mp4"
    save_path = tmp_path / "test.mp4"
    request_id = "Test-004"
    
    try:
        # 创建一个声明大小为1MB但实际只有512KB的响应
        mock_response = SizeMismatchResponse(status=200, content_length=TEST_FILE_SIZE_LARGE)
        mock_session = MockClientSession(mock_response)
        original_session = aiohttp.ClientSession
        aiohttp.ClientSession = lambda *args, **kwargs: mock_session
        
        result = await download_video(url, save_path, request_id)
        
        # 验证结果
        assert result == False, "大小不匹配时应该返回False"
        assert not save_path.exists(), "文件大小不匹配时应该删除文件"
        
        # 验证临时文件也被清理
        temp_path = Path(f"{save_path}.temp")
        assert not temp_path.exists(), "临时文件应该被清理"
        
    finally:
        aiohttp.ClientSession = original_session

@pytest.mark.asyncio
async def test_download_network_error(tmp_path):
    """测试网络错误的情况"""
    url = "http://example.com/test.mp4"
    save_path = tmp_path / "test.mp4"
    request_id = "Test-005"
    
    try:
        mock_response = MockResponse(
            status=200, 
            content_length=TEST_FILE_SIZE_LARGE,
            error="network"
        )
        mock_session = MockClientSession(mock_response)
        original_session = aiohttp.ClientSession
        aiohttp.ClientSession = lambda *args, **kwargs: mock_session
        
        result = await download_video(url, save_path, request_id)
        
        assert result == False
        assert not save_path.exists()
        
    finally:
        aiohttp.ClientSession = original_session

@pytest.mark.asyncio
async def test_download_error_logging(tmp_path, caplog):
    """测试错误情况下的日志记录"""
    caplog.set_level(logging.ERROR)
    
    url = "http://example.com/test.mp4"
    save_path = tmp_path / "test.mp4"
    request_id = "Test-Error"
    
    try:
        mock_response = ErrorResponse(error_type="network")
        mock_session = MockClientSession(mock_response)
        original_session = aiohttp.ClientSession
        aiohttp.ClientSession = lambda *args, **kwargs: mock_session
        
        result = await download_video(url, save_path, request_id)
        
        # 验证结果
        assert result == False
        assert not save_path.exists()
        
        # 验证错误日志
        error_logs = [r for r in caplog.records if r.levelname == 'ERROR']
        assert len(error_logs) > 0
        assert any("模拟的网络错误" in r.message for r in error_logs)
        
    finally:
        aiohttp.ClientSession = original_session

@pytest.mark.asyncio
async def test_download_progress_frequency(tmp_path, caplog):
    """测试进度日志的输出频率"""
    caplog.set_level(logging.INFO)
    
    url = "http://example.com/test.mp4"
    save_path = tmp_path / "test.mp4"
    request_id = "Test-Progress"
    
    try:
        # 使用较大的文件和延迟来确保看到进度日志
        mock_response = MockResponse(
            status=200, 
            content_length=TEST_FILE_SIZE_SMALL,  # 50MB
            delay=TEST_DELAY  # 1秒延迟
        )
        mock_session = MockClientSession(mock_response)
        original_session = aiohttp.ClientSession
        aiohttp.ClientSession = lambda *args, **kwargs: mock_session
        
        start_time = time.time()
        result = await download_video(url, save_path, request_id)
        end_time = time.time()
        
        # 验证结果
        assert result == True
        
        # 验证进度日志
        progress_logs = [r for r in caplog.records if "下载进度" in r.message]
        
        # 打印实际的日志数量和间隔
        print(f"\n总共有 {len(progress_logs)} 条进度日志")
        for i in range(1, len(progress_logs)):
            time_diff = progress_logs[i].created - progress_logs[i-1].created
            print(f"间隔 {i}: {time_diff:.2f}秒")
        
        # 验证日志数量和间隔
        assert len(progress_logs) >= 2, "应该至少有2条进度日志"  # 开始和结束
        assert len(progress_logs) <= 10, "进度日志不应该太频繁"   # 最多10条
        
        # 验证日志间隔
        if len(progress_logs) >= 2:
            for i in range(1, len(progress_logs)):
                time_diff = progress_logs[i].created - progress_logs[i-1].created
                assert time_diff >= 0.9, f"进度日志的间隔应该接近1秒，实际间隔: {time_diff:.2f}秒"
        
    finally:
        aiohttp.ClientSession = original_session