"""
Description: 物流数据索引器模块
    
-*- Encoding: UTF-8 -*-
@File     ：logistics_indexer.py.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午1:56
@Contact  ：king.songtao@gmail.com
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from es_storage.exceptions import ESIndexError
from es_storage.index_manager import ESIndexManager
from utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class LogisticsIndexer:
    """物流数据索引器"""

    def __init__(self, index_manager: ESIndexManager, text_processor: TextProcessor = None):
        """
        初始化索引器

        Args:
            index_manager: ES索引管理器
            text_processor: 文本处理器，如果为None则创建新实例
        """
        self.index_manager = index_manager
        self.text_processor = text_processor or TextProcessor()

    def prepare_track_document(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备物流轨迹文档

        Args:
            track_data: 原始轨迹数据

        Returns:
            Dict: ES文档
        """
        try:
            # 生成搜索文本
            search_text = self.text_processor.format_track_text(track_data)

            # 构建文档
            document = {
                "track_id": track_data.get("track_id"),
                "logistics_id": track_data.get("logistics_id"),
                "order_id": track_data.get("order_id"),
                "track_time": track_data.get("track_time"),
                "track_location": track_data.get("track_location"),
                "track_info": track_data.get("track_info"),
                "operator_name": track_data.get("operator_name"),
                "operator_phone": track_data.get("operator_phone"),
                "device_id": track_data.get("device_id"),
                "create_time": track_data.get("create_time"),
                "search_text": search_text
            }

            # 移除None值
            return {k: v for k, v in document.items() if v is not None}

        except Exception as e:
            logger.error(f"文档准备失败: {e}")
            raise ESIndexError(f"文档准备失败: {str(e)}")

    def index_tracks(self, tracks_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        索引物流轨迹数据

        Args:
            tracks_data: 轨迹数据列表

        Returns:
            Dict: 索引结果统计

        Raises:
            ESIndexError: 索引操作异常
        """
        try:
            if not tracks_data:
                logger.warning("没有需要索引的数据")
                return {"success": 0, "failed": 0}

            # 生成索引名称 (按月分表)
            current_date = datetime.now()
            index_name = f"logistics_tracks_{current_date.strftime('%Y%m')}"

            # 准备文档
            documents = []
            for track in tracks_data:
                try:
                    doc = self.prepare_track_document(track)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"处理轨迹数据失败: {e}, 数据: {track}")
                    continue

            # 批量索引
            if documents:
                result = self.index_manager.bulk_index_documents(index_name, documents)
                logger.info(f"索引完成: {result}")
                return result
            else:
                logger.warning("没有有效的文档需要索引")
                return {"success": 0, "failed": 0}

        except Exception as e:
            logger.error(f"索引轨迹数据失败: {e}")
            raise ESIndexError(f"索引轨迹数据失败: {str(e)}")
