import requests
import time

def submit_video_task(video_url: str) -> dict:
    """
    提交视频处理任务
    :param video_url: 视频URL
    :return: 任务ID
    """
    api_url = "http://localhost:8000/video-to-text"
    
    try:
        response = requests.post(
            api_url,
            json={"video_url": video_url}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "success": False,
            "error": f"提交任务失败: {str(e)}"
        }

def check_system_status() -> dict:
    """
    检查系统状态
    :return: 系统状态
    """
    api_url = "http://localhost:8000/system-status"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "success": False,
            "error": f"检查系统状态失败: {str(e)}"
        }