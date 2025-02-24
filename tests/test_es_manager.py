"""
ES管理模块测试脚本

-*- Encoding: UTF-8 -*-
@Author  : King Songtao
@Time    : 2024/2/24
"""

import pytest
import logging
from configs.es_config import ESConfig
from es_storage.connection_manager import ESConnectionManager
from es_storage.template_manager import ESTemplateManager
from es_storage.index_manager import ESIndexManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def es_config():
    """ES配置fixture"""
    return ESConfig(
        host="es-3omy2u5o.public.tencentelasticsearch.com",
        port=9200,
        username="elastic",
        password="Dst881009."
    )


@pytest.fixture(scope="module")
def conn_manager(es_config):
    """ES连接管理器fixture"""
    manager = ESConnectionManager(es_config)
    yield manager
    manager.close()


@pytest.fixture(scope="module")
def template_manager(conn_manager):
    """ES模板管理器fixture"""
    return ESTemplateManager(conn_manager)


@pytest.fixture(scope="module")
def index_manager(conn_manager):
    """ES索引管理器fixture"""
    return ESIndexManager(conn_manager)


def test_connection(conn_manager):
    """测试ES连接"""
    assert conn_manager.client.ping()
    logger.info("连接测试成功!")


def test_template_operations(template_manager):
    """测试模板操作"""
    test_template = {
        "index_patterns": ["test_*"],
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "test_field": {"type": "keyword"}
            }
        }
    }

    # 创建模板
    assert template_manager.create_template("test_template", test_template)
    logger.info("模板创建测试成功!")

    # 获取模板
    template = template_manager.get_template("test_template")
    assert template
    logger.info(f"获取模板测试成功: {template}")

    # 列出所有模板
    templates = template_manager.list_templates()
    assert "test_template" in templates
    logger.info(f"当前所有模板: {templates}")

    # 删除模板
    assert template_manager.delete_template("test_template")
    logger.info("模板删除测试成功!")


def test_index_operations(index_manager):
    """测试索引操作"""
    test_index = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "test_field": {"type": "keyword"}
            }
        }
    }

    index_name = "test_index"

    # 创建索引
    if not index_manager.index_exists(index_name):
        assert index_manager.create_index(index_name, test_index)
        logger.info("索引创建测试成功!")

    # 测试批量索引文档
    test_docs = [
        {"test_field": "value1"},
        {"test_field": "value2"}
    ]
    result = index_manager.bulk_index_documents(index_name, test_docs)
    assert result["success"] > 0
    assert result["failed"] == 0
    logger.info(f"批量索引测试成功: {result}")

    # 获取索引统计信息
    stats = index_manager.get_index_stats(index_name)
    assert stats
    logger.info(f"索引统计信息: {stats}")

    # 列出所有索引
    indices = index_manager.list_indices()
    assert index_name in indices
    logger.info(f"当前所有索引: {indices}")

    # 删除索引
    assert index_manager.delete_index(index_name)
    logger.info("索引删除测试成功!")
