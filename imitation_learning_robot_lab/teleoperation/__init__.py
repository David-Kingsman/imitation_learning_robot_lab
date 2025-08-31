from .handler import Handler
from .handler_factory import HandlerFactory

# from .joycon import *  # 暂时注释掉，避免pyjoycon模块缺失导致的导入错误
from .keyboard import *

HandlerFactory.register_all()
