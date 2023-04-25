from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from events import Event
import queue as q
from data_handlers import DataHandler


class Symbol(ABC):
    def __init__(self, name):
        self._name = name
        self._events: dict[q.Queue] = {}
        self._data_aggregators = []

    @property
    def name(self):
        return self._name

    def add_event(self, event: Event) -> None:
        try:
            self._events[event.type].put(event)
        except:
            self._events[event.type] = q.Queue()
            self._events[event.type].put(event)

    def get_event(self, type):
        return self._events[type].get()

    def add_data_aggergator(self, aggregator: DataHandler):
        self._data_aggregators.append(aggregator)

    @abstractmethod
    def aggregate_data(self):
        raise NotImplementedError("Implement Logic for aggregating data")
