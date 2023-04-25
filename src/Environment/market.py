from abc import ABC, ABCMeta
from exchanges import Exchange


class Market(ABC):
    def __init__(self, name):
        self._name = name
        self._exchanges = {}

    @property
    def name(self):
        return self._name

    def add_exchanges(self, exchange: Exchange):
        self._exchanges[exchange.name] = exchange
