"""
Description: CDC模块配置
    
-*- Encoding: UTF-8 -*-
@File     ：cdc_config.py
@Author   ：King Songtao
@Time     ：2025/2/23 下午11:30
@Contact  ：king.songtao@gmail.com
"""
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path


@dataclass
class CDCDatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 3306
    database: str = "logistics_db"
    user: str = "cdc_user"
    password: str = "cdc_password"
    charset: str = "utf8mb4"
    max_connections: int = 5


@dataclass
class CDCTableConfig:
    """表配置"""
    tables: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "orders": {
            "timestamp_column": "update_time",
            "primary_key": "order_id",
            "batch_size": 1000,
            "selected_columns": [
                "order_id", "order_status", "order_type",
                "create_time", "update_time", "total_amount"
            ]
        },
        "logistics_orders": {
            "timestamp_column": "update_time",
            "primary_key": "logistics_id",
            "batch_size": 1000,
            "selected_columns": [
                "logistics_id", "order_id", "logistics_status",
                "shipping_time", "create_time", "update_time"
            ]
        },
        "logistics_tracks": {
            "timestamp_column": "create_time",
            "primary_key": "track_id",
            "batch_size": 1000,
            "selected_columns": [
                "track_id", "logistics_id", "track_time",
                "track_location", "track_info", "create_time"
            ]
        }
    })


@dataclass
class CDCSchedulerConfig:
    """调度器配置"""
    interval_minutes: int = 30
    retry_times: int = 3
    retry_interval: int = 60
    output_dir: Path = Path("./data/logistics_knowledge")
    file_format: str = "json"
    compress: bool = True
    max_files_per_day: int = 48  # 每天最多文件数


@dataclass
class CDCConfig:
    """CDC总配置"""
    database: CDCDatabaseConfig = field(default_factory=CDCDatabaseConfig)
    tables: CDCTableConfig = field(default_factory=CDCTableConfig)
    scheduler: CDCSchedulerConfig = field(default_factory=CDCSchedulerConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "user": self.database.user,
                "password": "******",  # 密码脱敏
                "charset": self.database.charset,
                "max_connections": self.database.max_connections
            },
            "tables": self.tables.tables,
            "scheduler": {
                "interval_minutes": self.scheduler.interval_minutes,
                "retry_times": self.scheduler.retry_times,
                "retry_interval": self.scheduler.retry_interval,
                "output_dir": str(self.scheduler.output_dir),
                "file_format": self.scheduler.file_format,
                "compress": self.scheduler.compress,
                "max_files_per_day": self.scheduler.max_files_per_day
            }
        }


# 创建默认配置实例
cdc_config = CDCConfig()
