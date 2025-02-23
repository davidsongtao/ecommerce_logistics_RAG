"""
Description: 增量数据捕获使用示例

-*- Encoding: UTF-8 -*-
@File     ：cdc_example.py
@Author   ：King Songtao
@Time     ：2025/2/23
"""

import time
import signal
from pathlib import Path
from typing import Dict, List, Any
from configs.log_config import get_logger
from cdc_manager import CDCManager
from cdc_scheduler import CDCScheduler

logger = get_logger("cdc_example")


class LogisticsDataProcessor:
    """物流数据处理器"""

    @staticmethod
    def transform_to_knowledge_format(data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """将原始数据转换为知识库格式"""
        knowledge_items = []

        for track in data.get("knowledge_data", []):
            description = (
                f"订单 {track['order_id']} 在 {track['track_time']} "
                f"位于 {track['track_location']} 的物流状态: {track['track_info']}"
            )

            knowledge_item = {
                "id": f"TRACK_{track['track_id']}",
                "type": "logistics_track",
                "timestamp": track['create_time'],
                "title": f"物流轨迹 - {track['track_location']}",
                "content": description,
                "metadata": {
                    "order_id": track['order_id'],
                    "logistics_id": track['logistics_id'],
                    "order_status": track['order_status'],
                    "logistics_status": track['logistics_status'],
                    "track_location": track['track_location'],
                    "track_time": track['track_time']
                }
            }
            knowledge_items.append(knowledge_item)

        return knowledge_items

    @staticmethod
    def validate_data(knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证和清洗知识库数据"""
        valid_items = []

        for item in knowledge_items:
            # 验证必要字段
            required_fields = ["id", "type", "timestamp", "content"]
            if not all(field in item for field in required_fields):
                logger.warning(f"数据项缺少必要字段: {item.get('id', 'unknown')}")
                continue

            # 验证内容不为空
            if not item["content"].strip():
                logger.warning(f"数据项内容为空: {item['id']}")
                continue

            # 验证元数据完整性
            if "metadata" in item:
                required_metadata = [
                    "order_id", "logistics_id", "order_status",
                    "logistics_status", "track_location", "track_time"
                ]
                if not all(field in item["metadata"] for field in required_metadata):
                    logger.warning(f"数据项元数据不完整: {item['id']}")
                    continue

            valid_items.append(item)

        return valid_items


class LogisticsCDCService:
    """物流CDC服务"""

    def __init__(self, db_config: dict, output_dir: Path, interval_minutes: int = 30):
        self.cdc_manager = CDCManager(**db_config)
        self.scheduler = CDCScheduler(
            cdc_manager=self.cdc_manager,
            output_dir=output_dir,
            interval_minutes=interval_minutes
        )

        # 注册信号处理
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """处理关闭信号"""
        logger.info("接收到关闭信号，准备停止服务...")
        self.stop()

    def start(self):
        """启动服务"""
        try:
            logger.info("启动物流CDC服务...")
            self.scheduler.start()

            # 保持主进程运行
            while True:
                time.sleep(1)

        except Exception as e:
            logger.error(f"服务运行异常: {str(e)}")
            raise
        finally:
            self.stop()

    def stop(self):
        """停止服务"""
        self.scheduler.stop()
        self.cdc_manager.close()
        logger.info("物流CDC服务已停止")


def main():
    # 数据库配置
    db_config = {
        "host": "bj-cynosdbmysql-grp-4qfnlkl8.sql.tencentcdb.com",
        "port": 25597,
        "user": "root",
        "password": "Dst881009.",
        "database": "ecommerce_logistics"
    }

    # 指定输出目录
    output_dir = Path("./data/logistics_knowledge")

    # 创建并启动服务
    service = LogisticsCDCService(
        db_config=db_config,
        output_dir=output_dir,
        interval_minutes=1
    )

    service.start()


if __name__ == "__main__":
    main()
