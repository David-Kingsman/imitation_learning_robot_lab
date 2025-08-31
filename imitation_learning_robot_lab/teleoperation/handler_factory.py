from typing import Type

from .handler import Handler

from ..utils import ClassUtils


class HandlerFactory:
    _strategies = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[Handler]) -> None:
        cls._strategies[name] = strategy_cls

    @classmethod
    def get_strategies(cls, name: str) -> Type[Handler]:
        return cls._strategies.get(name)

    @classmethod
    def register_all(cls):
        leaf_subclasses: list[Type[Handler]] = ClassUtils.get_leaf_subclasses(Handler)
        for subclass in leaf_subclasses:
            cls.register(subclass.name, subclass)
