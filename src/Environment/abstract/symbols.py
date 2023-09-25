from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from events import Event
import queue as q
from data_handlers import Consumer, GatherStore


class Symbol(ABC, GatherStore):
    def __init__(self, symbol, exchange):
        super().__init__()
        self._symbol = symbol
        self._exchange = exchange
        self.latest_data = None

    @property
    def name(self):
        return self._name

    @property
    def exchange(self):
        return self._exchange

    def emit(self):
        for handler_type in self._data_handlers.keys():
            for handler_name in self._data_handlers[handler_type].keys():
                yield self._data_handlers[handler_type][handler_name].retrieve_latest()

    @property
    def events_type(self):
        return set(self._events.keys())

