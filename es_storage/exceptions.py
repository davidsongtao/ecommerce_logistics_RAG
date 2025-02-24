"""
Description: ES自定义异常模块
    
-*- Encoding: UTF-8 -*-
@File     ：exceptions.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午1:40
@Contact  ：king.songtao@gmail.com
"""


class ESError(Exception):
    """ES基础异常"""

    def __init__(self, message: str = None, status_code: int = None):
        self.message = message or "ES操作发生未知错误"
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self):
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class ESConnectionError(ESError):
    """ES连接异常"""

    def __init__(self, message: str = None, status_code: int = None):
        super().__init__(
            message or "ES连接失败",
            status_code or 503
        )


class ESIndexError(ESError):
    """ES索引操作异常"""

    def __init__(self, message: str = None, index_name: str = None, status_code: int = None):
        self.index_name = index_name
        message = message or f"索引 {index_name} 操作失败"
        super().__init__(message, status_code or 400)

    def __str__(self):
        return f"{super().__str__()} [索引: {self.index_name}]"


class ESTemplateError(ESError):
    """ES模板操作异常"""

    def __init__(self, message: str = None, template_name: str = None, status_code: int = None):
        self.template_name = template_name
        message = message or f"模板 {template_name} 操作失败"
        super().__init__(message, status_code or 400)

    def __str__(self):
        return f"{super().__str__()} [模板: {self.template_name}]"


class ESQueryError(ESError):
    """ES查询异常"""

    def __init__(self, message: str = None, query: dict = None, status_code: int = None):
        self.query = query
        message = message or "查询执行失败"
        super().__init__(message, status_code or 400)

    def __str__(self):
        query_str = str(self.query) if self.query else "未知查询"
        return f"{super().__str__()} [查询: {query_str}]"


class ESBulkError(ESError):
    """ES批量操作异常"""

    def __init__(self, message: str = None, failed_items: list = None, status_code: int = None):
        self.failed_items = failed_items or []
        message = message or f"批量操作失败 ({len(self.failed_items)}个项目)"
        super().__init__(message, status_code or 400)

    def __str__(self):
        return f"{super().__str__()} [失败项目数: {len(self.failed_items)}]"


class ESConfigError(ESError):
    """ES配置异常"""

    def __init__(self, message: str = None, config_key: str = None, status_code: int = None):
        self.config_key = config_key
        message = message or f"配置项 {config_key} 无效"
        super().__init__(message, status_code or 400)

    def __str__(self):
        return f"{super().__str__()} [配置项: {self.config_key}]"


class ESTimeoutError(ESError):
    """ES超时异常"""

    def __init__(self, message: str = None, timeout: float = None, status_code: int = None):
        self.timeout = timeout
        message = message or f"操作超时 ({timeout}秒)"
        super().__init__(message, status_code or 408)

    def __str__(self):
        return f"{super().__str__()} [超时: {self.timeout}秒]"
