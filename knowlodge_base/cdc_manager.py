"""
Description: 基于时间戳的增量数据捕获模块
    
-*- Encoding: UTF-8 -*-
@File     ：cdc_manager.py
@Author   ：King Songtao
@Time     ：2025/2/23 下午11:29
@Contact  ：king.songtao@gmail.com
"""
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pymysql
from pymysql.cursors import DictCursor
from configs.config import config
from configs.log_config import get_logger

logger = get_logger("cdc_manager")


@dataclass
class TableConfig:
    """表配置"""
    table_name: str
    timestamp_column: str
    primary_key: str
    selected_columns: List[str]
    batch_size: int = 1000


class CDCManager:
    """增量数据捕获管理器"""

    def __init__(
            self,
            host: str,
            port: int,
            user: str,
            password: str,
            database: str
    ):
        self.db_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": "utf8mb4",
            "cursorclass": DictCursor
        }

        # 初始化数据库连接
        self.conn = self._get_connection()

        # 初始化监控表配置
        self.table_configs = self._init_table_configs()

    def _get_connection(self) -> pymysql.Connection:
        """获取数据库连接"""
        try:
            return pymysql.connect(**self.db_config)
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise

    def _init_table_configs(self) -> Dict[str, TableConfig]:
        """初始化表配置"""
        return {
            # 订单主表
            "orders": TableConfig(
                table_name="orders",
                timestamp_column="create_time",
                primary_key="order_id",
                selected_columns=[
                    "order_id", "order_status", "order_type",
                    "create_time", "total_amount"
                ]
            ),
            # 物流订单表
            "logistics_orders": TableConfig(
                table_name="logistics_orders",
                timestamp_column="update_time",
                primary_key="logistics_id",
                selected_columns=[
                    "logistics_id", "order_id", "logistics_status",
                    "shipping_time", "create_time", "update_time"
                ]
            ),
            # 物流轨迹表
            "logistics_tracks": TableConfig(
                table_name="logistics_tracks",
                timestamp_column="create_time",
                primary_key="track_id",
                selected_columns=[
                    "track_id", "logistics_id", "track_time",
                    "track_location", "track_info", "create_time"
                ]
            )
        }

    def capture_incremental_data(
            self,
            table_name: str,
            start_time: datetime.datetime,
            end_time: Optional[datetime.datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        捕获增量数据
        """
        if table_name not in self.table_configs:
            raise ValueError(f"未配置的表名: {table_name}")

        table_config = self.table_configs[table_name]
        end_time = end_time or datetime.datetime.now()

        try:
            with self.conn.cursor() as cursor:
                # 构建SQL
                columns = ", ".join(table_config.selected_columns)
                sql = f"""
                    SELECT {columns}
                    FROM {table_config.table_name}
                    WHERE {table_config.timestamp_column} >= %s
                    AND {table_config.timestamp_column} <= %s
                    ORDER BY {table_config.timestamp_column}
                """

                # 打印调试信息
                logger.info(f"捕获 {table_name} 表增量数据 SQL 参数："
                            f"开始时间：{start_time}，"
                            f"结束时间：{end_time}")

                # 分批查询
                results = []
                offset = 0
                while True:
                    batch_sql = f"{sql} LIMIT {offset}, {table_config.batch_size}"
                    cursor.execute(batch_sql, (start_time, end_time))
                    batch_data = cursor.fetchall()

                    if not batch_data:
                        break

                    results.extend(batch_data)
                    offset += table_config.batch_size

                    logger.info(f"已获取{table_name}表增量数据: {len(results)}条")

                return results

        except Exception as e:
            logger.error(f"捕获增量数据失败: {str(e)}")
            raise

    def process_logistics_data(
            self,
            start_time: datetime.datetime,
            end_time: Optional[datetime.datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理物流相关的增量数据
        """
        try:
            # 获取各表增量数据
            orders_data = self.capture_incremental_data("orders", start_time, end_time)
            logistics_data = self.capture_incremental_data("logistics_orders", start_time, end_time)
            tracks_data = self.capture_incremental_data("logistics_tracks", start_time, end_time)

            # 数据关联逻辑优化
            knowledge_data = []

            # 构建快速查找字典
            logistics_dict = {log["logistics_id"]: log for log in logistics_data}
            order_dict = {order["order_id"]: order for order in orders_data}

            # 1. 处理轨迹表数据
            if tracks_data:
                for track in tracks_data:
                    knowledge_item = {
                        "track_id": track["track_id"],
                        "track_time": track["track_time"],
                        "track_location": track["track_location"],
                        "track_info": track["track_info"],
                        "create_time": track["create_time"],
                        "data_type": "track_only"
                    }

                    # 尝试关联物流订单
                    logistics_id = track["logistics_id"]
                    if logistics_id in logistics_dict:
                        logistics = logistics_dict[logistics_id]
                        knowledge_item.update({
                            "logistics_id": logistics_id,
                            "logistics_status": logistics["logistics_status"]
                        })

                        # 尝试关联主订单
                        order_id = logistics["order_id"]
                        if order_id in order_dict:
                            order = order_dict[order_id]
                            knowledge_item.update({
                                "order_id": order_id,
                                "order_status": order["order_status"]
                            })

                    knowledge_data.append(knowledge_item)

            # 2. 处理物流订单数据
            if logistics_data:
                for logistics in logistics_data:
                    knowledge_item = {
                        "logistics_id": logistics["logistics_id"],
                        "shipping_time": logistics.get("shipping_time"),
                        "logistics_status": logistics["logistics_status"],
                        "create_time": logistics["create_time"],
                        "data_type": "logistics_only"
                    }

                    # 尝试关联主订单
                    order_id = logistics["order_id"]
                    if order_id in order_dict:
                        order = order_dict[order_id]
                        knowledge_item.update({
                            "order_id": order_id,
                            "order_status": order["order_status"]
                        })

                    knowledge_data.append(knowledge_item)

            # 3. 处理订单表数据
            if orders_data:
                for order in orders_data:
                    knowledge_item = {
                        "order_id": order["order_id"],
                        "order_status": order["order_status"],
                        "order_type": order.get("order_type", "未知"),
                        "total_amount": order.get("total_amount", 0),
                        "create_time": order["create_time"],
                        "data_type": "order_only"
                    }

                    # 尝试关联物流订单
                    related_logistics = [
                        log for log in logistics_data
                        if log["order_id"] == order["order_id"]
                    ]
                    if related_logistics:
                        knowledge_item.update({
                            "logistics_id": related_logistics[0]["logistics_id"],
                            "logistics_status": related_logistics[0]["logistics_status"]
                        })

                    knowledge_data.append(knowledge_item)

            # 记录日志
            logger.info(f"处理增量数据："
                        f"订单数：{len(orders_data)}，"
                        f"物流订单数：{len(logistics_data)}，"
                        f"物流轨迹数：{len(tracks_data)}，"
                        f"知识数据数：{len(knowledge_data)}")

            return {
                "orders": orders_data,
                "logistics_orders": logistics_data,
                "logistics_tracks": tracks_data,
                "knowledge_data": knowledge_data
            }

        except Exception as e:
            logger.error(f"处理物流数据失败: {str(e)}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
