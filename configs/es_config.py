"""
Description: ES配置管理模块
    
-*- Encoding: UTF-8 -*-
@File     ：es_config.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午1:34
@Contact  ：king.songtao@gmail.com
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ESConfig:
    """ES配置类"""
    host: str
    port: int
    username: str
    password: str
    use_ssl: bool = True
    verify_certs: bool = False
    timeout: int = 30
    max_retries: int = 3

    # 索引配置
    index_settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.index_settings is None:
            self.index_settings = {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "refresh_interval": "30s",
                "analysis": {
                    "analyzer": {
                        "ik_smart_analyzer": {
                            "type": "custom",
                            "tokenizer": "ik_smart",
                            "filter": ["lowercase", "trim"]
                        }
                    }
                }
            }

    @property
    def connection_params(self) -> Dict[str, Any]:
        """获取连接参数"""
        return {
            "hosts": [f"https://{self.host}:{self.port}"],
            "http_auth": (self.username, self.password),
            "verify_certs": self.verify_certs,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
