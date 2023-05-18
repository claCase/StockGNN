from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from events import Event
import queue as q
from data_handlers import DataHandler, GatherStore


class Symbol(ABC, GatherStore):
    def __init__(self, name):
        super().__init__()
        self._name = name
        self.latest_data = None

    @property
    def name(self):
        return self._name

    def emit(self, type):
        for handler_type in self._data_handlers.keys():
            for handler_name in self._data_handlers[handler_type].keys():
                latest_handler_data = self._data_handlers[handler_type][handler_name].get
                return self._events[type].get()

    @property
    def events_type(self):
        return set(self._events.keys())
