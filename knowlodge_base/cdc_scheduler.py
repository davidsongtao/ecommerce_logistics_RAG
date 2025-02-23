"""
Description: 增量数据捕获调度器
    
-*- Encoding: UTF-8 -*-
@File     ：cdc_scheduler.py
@Author   ：King Songtao
@Time     ：2025/2/23 下午11:33
@Contact  ：king.songtao@gmail.com
"""
import time
import datetime
import json
from pathlib import Path
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from configs.config import config
from configs.log_config import get_logger
from cdc_manager import CDCManager

logger = get_logger("cdc_scheduler")


class CDCScheduler:
    """增量数据捕获调度器"""

    def __init__(
            self,
            cdc_manager: CDCManager,
            output_dir: Optional[Path] = None,
            interval_minutes: int = 30
    ):
        self.cdc_manager = cdc_manager
        self.output_dir = output_dir or config.project_root / "data" / "cdc"
        self.interval_minutes = interval_minutes

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化调度器
        self.scheduler = BackgroundScheduler()

        # 记录上次执行时间
        self.last_exec_time_file = self.output_dir / "last_exec_time.txt"
        self.last_exec_time = self._load_last_exec_time()

    def _load_last_exec_time(self) -> datetime.datetime:
        """加载上次执行时间"""
        try:
            if self.last_exec_time_file.exists():
                timestamp = float(self.last_exec_time_file.read_text())
                return datetime.datetime.fromtimestamp(timestamp)
        except Exception as e:
            logger.warning(f"加载上次执行时间失败: {str(e)}")

        # 默认为当前时间前30分钟
        return datetime.datetime.now() - datetime.timedelta(minutes=self.interval_minutes)

    def _save_last_exec_time(self, exec_time: datetime.datetime):
        """保存执行时间"""
        try:
            self.last_exec_time_file.write_text(str(exec_time.timestamp()))
        except Exception as e:
            logger.error(f"保存执行时间失败: {str(e)}")

    def _save_incremental_data(self, data: dict, timestamp: datetime.datetime):
        """保存增量数据"""
        try:
            # 检查是否有有效数据
            has_data = any(len(data_list) > 0 for data_list in data.values())

            if not has_data:
                logger.info(f"在 {timestamp} 未找到增量数据，跳过保存")
                return

            # 按日期组织目录
            date_dir = self.output_dir / timestamp.strftime("%Y-%m-%d")
            date_dir.mkdir(exist_ok=True)

            # 保存文件
            file_name = f"cdc_data_{timestamp.strftime('%H%M%S')}.json"
            file_path = date_dir / file_name

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, default=str, indent=2)

            logger.info(f"增量数据已保存: {file_path}")

        except Exception as e:
            logger.error(f"保存增量数据失败: {str(e)}")
            raise

    def execute_capture(self):
        """执行数据捕获"""
        try:
            current_time = datetime.datetime.now()
            logger.info(f"开始执行增量数据捕获: {self.last_exec_time} -> {current_time}")

            # 获取增量数据
            data = self.cdc_manager.process_logistics_data(
                start_time=self.last_exec_time,
                end_time=current_time
            )

            # 打印详细的数据信息
            for key, value in data.items():
                logger.info(f"{key} 数据条数: {len(value)}")

            # 保存数据
            self._save_incremental_data(data, current_time)

            # 更新执行时间
            self._save_last_exec_time(current_time)
            self.last_exec_time = current_time

            logger.info("增量数据捕获完成")

        except Exception as e:
            logger.error(f"执行增量数据捕获失败: {str(e)}")

    def start(self):
        """启动调度器"""
        try:
            # 添加定时任务
            trigger = CronTrigger(
                minute=f"*/{self.interval_minutes}",
                timezone="Asia/Shanghai"
            )

            self.scheduler.add_job(
                self.execute_capture,
                trigger=trigger,
                id="cdc_capture"
            )

            # 启动调度器
            self.scheduler.start()
            logger.info(f"调度器已启动, 间隔时间: {self.interval_minutes}分钟")

        except Exception as e:
            logger.error(f"启动调度器失败: {str(e)}")
            raise

    def stop(self):
        """停止调度器"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("调度器已停止")
