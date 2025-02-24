"""
Description: 
    
-*- Encoding: UTF-8 -*-
@File     ：connection_manager.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午1:35
@Contact  ：king.songtao@gmail.com
"""
"""
ES连接管理模块
"""
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError
import logging
from typing import Optional
from configs.es_config import ESConfig
from es_storage.exceptions import ESConnectionError

logger = logging.getLogger(__name__)


class ESConnectionManager:
    """ES连接管理器"""

    def __init__(self, config: ESConfig):
        self.config = config
        self._client: Optional[Elasticsearch] = None

    @property
    def client(self) -> Elasticsearch:
        """获取ES客户端"""
        if self._client is None:
            self.connect()
        return self._client

    def connect(self) -> None:
        """建立连接"""
        try:
            self._client = Elasticsearch(**self.config.connection_params)
            if not self._client.ping():
                raise ESConnectionError("无法连接到ES集群")
            logger.info("成功连接到ES集群")
        except ConnectionError as e:
            logger.error(f"ES连接失败: {str(e)}")
            raise ESConnectionError(f"ES连接失败: {str(e)}")

    def close(self) -> None:
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("ES连接已关闭")
