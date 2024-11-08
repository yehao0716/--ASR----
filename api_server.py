from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from video_processor import process_video_url_async
import time
from datetime import datetime
from utils.logger import setup_logger
from utils.resource_monitor import ResourceMonitor
from utils.resource_manager import ResourceManager, ResourceThreshold
import os
from config import Config
import asyncio

# 确保所需目录存在
Config.setup()

# 设置日志
logger = setup_logger()

# 创建资源监控器
resource_monitor = ResourceMonitor(log_interval=5)

# 创建资源管理器
resource_manager = ResourceManager(
    ResourceThreshold(
        max_cpu_percent=Config.CPU_THRESHOLD,
        max_memory_percent=Config.MEMORY_THRESHOLD,
        max_gpu_percent=Config.GPU_THRESHOLD,
        max_concurrent_tasks=Config.MAX_CONCURRENT_TASKS
    )
)

app = FastAPI(
    title="视频ASR文本转述API",
    description="将视频转换为文字内容的API服务"
)

class VideoRequest(BaseModel):
    video_url: str
    proxy: str = None

@app.post("/video-to-text")
async def video_to_text(request: VideoRequest):
    start_time = time.time()
    request_id = f"Request-{int(start_time)}"
    
    logger.info(f"[{request_id}] 收到新的视频处理请求: {request.video_url}")
    
    try:
        # 检查资源是否足够
        logger.info(f"[{request_id}] 开始检查资源...")
        resources_available, reason = await resource_manager.check_resources()
        if not resources_available:
            logger.warning(f"[{request_id}] 资源不足: {reason}")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "系统资源不足，请稍后重试",
                    "reason": reason
                }
            )
        
        # 异步获取资源
        logger.info(f"[{request_id}] 尝试获取资源...")
        acquire_result = await resource_manager.acquire_resource()
        logger.info(f"[{request_id}] 资源获取结果: {acquire_result}")
        
        if acquire_result:
            try:
                logger.info(f"[{request_id}] 资源获取成功，准备开始处理视频...")
                # 处理视频，传递request_id
                result = await process_video_url_async(request.video_url, request_id)  # 添加request_id参数
                logger.info(f"[{request_id}] process_video_url_async 返回结果: {result}")
                
                end_time = time.time()
                process_time = end_time - start_time
                
                logger.info(f"[{request_id}] 处理完成，耗时: {process_time:.2f}秒")
                
                return {
                    "request_id": request_id,
                    "process_time": f"{process_time:.2f}秒",
                    "success": result.get("success", False),
                    "text": result.get("text", ""),
                    "error": result.get("error", None)
                }
            except Exception as e:
                logger.error(f"[{request_id}] 视频处理过程发生异常: {str(e)}")
                raise
            finally:
                # 确保释放资源
                logger.info(f"[{request_id}] 准备释放资源...")
                await resource_manager.release_resource()
                logger.info(f"[{request_id}] 资源已释放")
        else:
            logger.error(f"[{request_id}] 无法获取资源")
            raise HTTPException(
                status_code=503,
                detail="无法获取系统资源，请稍后重试"
            )
            
    except Exception as e:
        error_msg = f"处理异常: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        return {
            "request_id": request_id,
            "process_time": f"{time.time() - start_time:.2f}秒",
            "success": False,
            "text": "",
            "error": error_msg
        }

@app.get("/system-status")
async def get_system_status():
    """获取系统资源状态"""
    resources_available, reason = await resource_manager.check_resources()
    return {
        "resources_available": resources_available,
        "reason": reason,
        "current_tasks": resource_manager.current_tasks,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("启动视频ASR文本转述API服务...")
    resource_monitor.start()
    try:
        uvicorn.run(app, host=Config.HOST, port=Config.PORT)
    finally:
        resource_monitor.stop()