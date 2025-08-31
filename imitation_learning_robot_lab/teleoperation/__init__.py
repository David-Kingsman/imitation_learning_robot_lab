from .handler import Handler
from .handler_factory import HandlerFactory

from .joycon import *
from .keyboard import *

HandlerFactory.register_all()
