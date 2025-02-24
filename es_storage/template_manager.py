"""
Description: 
    
-*- Encoding: UTF-8 -*-
@File     ：template_manager.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午1:44
@Contact  ：king.songtao@gmail.com
"""
import logging
from typing import Dict, Any, List
from elasticsearch import NotFoundError
from .exceptions import ESTemplateError
from .connection_manager import ESConnectionManager

logger = logging.getLogger(__name__)


class ESTemplateManager:
    """ES模板管理器"""

    def __init__(self, connection_manager: ESConnectionManager):
        self.conn_manager = connection_manager
        self.client = connection_manager.client

    def create_template(self, name: str, body: Dict[str, Any]) -> bool:
        """
        创建索引模板

        Args:
            name: 模板名称
            body: 模板配置

        Returns:
            bool: 是否创建成功

        Raises:
            ESTemplateError: 模板操作异常
        """
        try:
            self.client.indices.put_template(name=name, body=body)
            logger.info(f"成功创建索引模板: {name}")
            return True
        except Exception as e:
            logger.error(f"创建索引模板失败: {str(e)}")
            raise ESTemplateError(f"创建模板失败: {str(e)}", template_name=name)

    def get_template(self, name: str) -> Dict[str, Any]:
        """
        获取索引模板

        Args:
            name: 模板名称

        Returns:
            Dict: 模板配置

        Raises:
            ESTemplateError: 模板操作异常
        """
        try:
            return self.client.indices.get_template(name=name)
        except NotFoundError:
            logger.warning(f"模板不存在: {name}")
            return {}
        except Exception as e:
            logger.error(f"获取索引模板失败: {str(e)}")
            raise ESTemplateError(f"获取模板失败: {str(e)}", template_name=name)

    def delete_template(self, name: str) -> bool:
        """
        删除索引模板

        Args:
            name: 模板名称

        Returns:
            bool: 是否删除成功

        Raises:
            ESTemplateError: 模板操作异常
        """
        try:
            self.client.indices.delete_template(name=name)
            logger.info(f"成功删除索引模板: {name}")
            return True
        except NotFoundError:
            logger.warning(f"要删除的模板不存在: {name}")
            return False
        except Exception as e:
            logger.error(f"删除索引模板失败: {str(e)}")
            raise ESTemplateError(f"删除模板失败: {str(e)}", template_name=name)

    def list_templates(self) -> List[str]:
        """
        获取所有模板名称列表

        Returns:
            List[str]: 模板名称列表

        Raises:
            ESTemplateError: 模板操作异常
        """
        try:
            templates = self.client.indices.get_template()
            return list(templates.keys())
        except Exception as e:
            logger.error(f"获取模板列表失败: {str(e)}")
            raise ESTemplateError(f"获取模板列表失败: {str(e)}")

    def template_exists(self, name: str) -> bool:
        """
        检查模板是否存在

        Args:
            name: 模板名称

        Returns:
            bool: 是否存在
        """
        try:
            return self.client.indices.exists_template(name=name)
        except Exception as e:
            logger.error(f"检查模板存在失败: {str(e)}")
            return False

    def get_template_mapping(self, name: str) -> Dict[str, Any]:
        """
        获取模板的mapping配置

        Args:
            name: 模板名称

        Returns:
            Dict: mapping配置
        """
        template = self.get_template(name)
        return template.get(name, {}).get('mappings', {})

    def update_template(self, name: str, body: Dict[str, Any], create_if_not_exists: bool = True) -> bool:
        """
        更新索引模板

        Args:
            name: 模板名称
            body: 新的模板配置
            create_if_not_exists: 如果模板不存在是否创建

        Returns:
            bool: 是否更新成功
        """
        try:
            if not self.template_exists(name):
                if create_if_not_exists:
                    return self.create_template(name, body)
                else:
                    raise ESTemplateError(f"模板不存在: {name}", template_name=name)

            self.client.indices.put_template(name=name, body=body)
            logger.info(f"成功更新索引模板: {name}")
            return True

        except Exception as e:
            logger.error(f"更新索引模板失败: {str(e)}")
            raise ESTemplateError(f"更新模板失败: {str(e)}", template_name=name)
