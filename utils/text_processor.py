"""
Description: 将MySQL数据库中的数据转换成自然语言
    
-*- Encoding: UTF-8 -*-
@File     ：text_processor.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午1:54
@Contact  ：king.songtao@gmail.com
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TextProcessor:
    """文本处理工具类"""

    @staticmethod
    def format_datetime(dt_value: Any) -> str:
        """
        格式化日期时间

        Args:
            dt_value: 日期时间值(字符串或datetime对象)

        Returns:
            str: 格式化后的时间字符串
        """
        try:
            if isinstance(dt_value, str):
                # 尝试解析字符串为datetime
                dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(dt_value, datetime):
                return dt_value.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return '未知时间'
        except Exception as e:
            logger.warning(f"时间格式化失败: {e}")
            return '未知时间'

    @staticmethod
    def format_track_text(track_data: Dict[str, Any]) -> str:
        """
        将物流轨迹数据转换为自然语言文本

        Args:
            track_data: 轨迹数据字典

        Returns:
            str: 格式化的文本
        """
        try:
            # 处理必要字段
            order_id = track_data.get('order_id', '未知')
            time_str = TextProcessor.format_datetime(track_data.get('track_time'))
            location = track_data.get('track_location', '未知位置')
            track_info = track_data.get('track_info', '')

            # 构建基础文本
            text_parts = [
                f"订单{order_id}的包裹",
                f"于{time_str}",
                f"在{location}"
            ]

            # 添加操作员信息
            if operator_name := track_data.get('operator_name'):
                text_parts.append(f"由{operator_name}")

            # 添加轨迹信息
            if track_info:
                text_parts.append(track_info)

            return "，".join(text_parts)

        except Exception as e:
            logger.error(f"轨迹文本格式化失败: {e}")
            return f"订单{track_data.get('order_id', '未知')}的轨迹信息处理失败"
