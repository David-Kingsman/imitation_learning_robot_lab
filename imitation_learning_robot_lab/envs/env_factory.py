from typing import Type

from .env import Env

from ..utils import ClassUtils


class EnvFactory:
    _strategies = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[Env]) -> None:
        cls._strategies[name] = strategy_cls

    @classmethod
    def get_strategies(cls, name: str) -> Type[Env]:
        return cls._strategies.get(name)

    @classmethod
    def register_all(cls):
        leaf_subclasses: list[Type[Env]] = ClassUtils.get_leaf_subclasses(Env)
        for subclass in leaf_subclasses:
            cls.register(subclass.name, subclass)
