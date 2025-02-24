"""
Description: ES索引管理模块
    
-*- Encoding: UTF-8 -*-
@File     ：index_manager.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午1:46
@Contact  ：king.songtao@gmail.com
"""

import logging
from typing import Dict, Any, List, Optional
from elasticsearch import NotFoundError, ConflictError
from elasticsearch.helpers import bulk
from datetime import datetime
from .exceptions import ESIndexError, ESBulkError
from .connection_manager import ESConnectionManager

logger = logging.getLogger(__name__)


class ESIndexManager:
    """ES索引管理器"""

    def __init__(self, connection_manager: ESConnectionManager):
        self.conn_manager = connection_manager
        self.client = connection_manager.client

    def create_index(self, index_name: str, body: Dict[str, Any]) -> bool:
        """
        创建索引

        Args:
            index_name: 索引名称
            body: 索引配置

        Returns:
            bool: 是否创建成功

        Raises:
            ESIndexError: 索引操作异常
        """
        try:
            if self.index_exists(index_name):
                raise ESIndexError(f"索引已存在", index_name=index_name)

            self.client.indices.create(index=index_name, body=body)
            logger.info(f"成功创建索引: {index_name}")
            return True

        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            raise ESIndexError(f"创建索引失败: {str(e)}", index_name=index_name)

    def delete_index(self, index_name: str) -> bool:
        """
        删除索引

        Args:
            index_name: 索引名称

        Returns:
            bool: 是否删除成功
        """
        try:
            if not self.index_exists(index_name):
                logger.warning(f"要删除的索引不存在: {index_name}")
                return False

            self.client.indices.delete(index=index_name)
            logger.info(f"成功删除索引: {index_name}")
            return True

        except Exception as e:
            logger.error(f"删除索引失败: {str(e)}")
            raise ESIndexError(f"删除索引失败: {str(e)}", index_name=index_name)

    def index_exists(self, index_name: str) -> bool:
        """
        检查索引是否存在

        Args:
            index_name: 索引名称

        Returns:
            bool: 是否存在
        """
        try:
            return self.client.indices.exists(index=index_name)
        except Exception as e:
            logger.error(f"检查索引存在失败: {str(e)}")
            return False

    def get_mapping(self, index_name: str) -> Dict[str, Any]:
        """
        获取索引的mapping

        Args:
            index_name: 索引名称

        Returns:
            Dict: mapping配置
        """
        try:
            return self.client.indices.get_mapping(index=index_name)
        except NotFoundError:
            logger.warning(f"索引不存在: {index_name}")
            return {}
        except Exception as e:
            logger.error(f"获取mapping失败: {str(e)}")
            raise ESIndexError(f"获取mapping失败: {str(e)}", index_name=index_name)

    def bulk_index_documents(
            self,
            index_name: str,
            documents: List[Dict[str, Any]],
            chunk_size: int = 500
    ) -> Dict[str, int]:
        """
        批量索引文档

        Args:
            index_name: 索引名称
            documents: 文档列表
            chunk_size: 批次大小

        Returns:
            Dict: 操作结果统计
        """
        try:
            # 准备批量操作
            actions = [
                {
                    "_index": index_name,
                    "_source": doc
                }
                for doc in documents
            ]

            # 执行批量操作
            success, failed = bulk(
                self.client,
                actions,
                chunk_size=chunk_size,
                raise_on_error=False,
                stats_only=True
            )

            logger.info(f"批量索引完成 - 成功: {success}, 失败: {failed}")

            if failed > 0:
                raise ESBulkError(
                    f"部分文档索引失败: {failed}/{success + failed}",
                    failed_items=failed
                )

            return {"success": success, "failed": failed}

        except Exception as e:
            logger.error(f"批量索引失败: {str(e)}")
            raise ESBulkError(f"批量索引失败: {str(e)}")

    def get_index_settings(self, index_name: str) -> Dict[str, Any]:
        """
        获取索引设置

        Args:
            index_name: 索引名称

        Returns:
            Dict: 索引设置
        """
        try:
            return self.client.indices.get_settings(index=index_name)
        except NotFoundError:
            logger.warning(f"索引不存在: {index_name}")
            return {}
        except Exception as e:
            logger.error(f"获取索引设置失败: {str(e)}")
            raise ESIndexError(f"获取索引设置失败: {str(e)}", index_name=index_name)

    def update_settings(self, index_name: str, settings: Dict[str, Any]) -> bool:
        """
        更新索引设置

        Args:
            index_name: 索引名称
            settings: 新的设置

        Returns:
            bool: 是否更新成功
        """
        try:
            self.client.indices.put_settings(index=index_name, body=settings)
            logger.info(f"成功更新索引设置: {index_name}")
            return True
        except Exception as e:
            logger.error(f"更新索引设置失败: {str(e)}")
            raise ESIndexError(f"更新索引设置失败: {str(e)}", index_name=index_name)

    def refresh_index(self, index_name: str) -> None:
        """
        刷新索引

        Args:
            index_name: 索引名称
        """
        try:
            self.client.indices.refresh(index=index_name)
            logger.debug(f"索引刷新完成: {index_name}")
        except Exception as e:
            logger.error(f"索引刷新失败: {str(e)}")
            raise ESIndexError(f"索引刷新失败: {str(e)}", index_name=index_name)

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        获取索引统计信息

        Args:
            index_name: 索引名称

        Returns:
            Dict: 统计信息
        """
        try:
            return self.client.indices.stats(index=index_name)
        except Exception as e:
            logger.error(f"获取索引统计信息失败: {str(e)}")
            raise ESIndexError(f"获取索引统计信息失败: {str(e)}", index_name=index_name)

    def list_indices(self, pattern: str = "*") -> List[str]:
        """
        列出所有索引

        Args:
            pattern: 索引名称匹配模式

        Returns:
            List[str]: 索引名称列表
        """
        try:
            return list(self.client.indices.get_alias(pattern).keys())
        except Exception as e:
            logger.error(f"获取索引列表失败: {str(e)}")
            raise ESIndexError(f"获取索引列表失败: {str(e)}")
