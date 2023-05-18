from abc import ABC, abstractmethod
from events import Event
from queue import Queue
from typing import Mapping


class DataHandler(ABC):
    def __init__(self, type, name="default"):
        self._type = type
        self._name = name
        self._latest_data = Queue()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @abstractmethod
    def _gather_data(self) -> [Event]:
        """
        Gathers latest data from either APIs, DataBases or Local Files
        """
        raise NotImplementedError

    def gather_data(self):
        self._latest_data.put(self._gather_data())

    @abstractmethod
    def _store_data(self, data):
        """
        Stores the latest data items in Local Files or DataBase
        """
        raise NotImplementedError

    def store_data(self, data):
        self._store_data(data)

    @abstractmethod
    def retrieve_data(self, args):
        """
        Retrieves data based on args, which would mainly be time intervals
        """
        raise NotImplementedError

    def retrieve_latest(self):
        """
        Retrieves last event
        """
        if self._latest_data.empty():
            print("Empty Queue, not data to emit")
        else:
            return self._latest_data.get()


class GatherStore(ABC):
    def __init__(self):
        self._data_handlers: Mapping[str:Mapping[str:DataHandler]] = {}

    def add_data_handler(self, data_handler: DataHandler):
        if data_handler.type in self._data_handlers.keys():
            print(f"Adding data handler of type {data_handler.type}")
            if data_handler.name not in self._data_handlers[data_handler.type].keys():
                print(f"Adding data handler of type {data_handler.type} with {data_handler.name}")
                self._data_handlers[data_handler.type][data_handler.name] = data_handler
            else:
                raise KeyError(
                    f"Handler of type {data_handler.type} with name {data_handler.name} has been already initialized")
        else:
            print(f"Initializing of type {data_handler.type} with name {data_handler.name}")
            self._data_handlers[data_handler.type] = {}
            self._data_handlers[data_handler.type][data_handler.name] = data_handler

    def get_data_handlers(self):
        return self._data_handlers

    def retrieve_handler_data(self, type, name, args):
        try:
            return self._data_handlers[type][name].retieve_data(args)
        except Exception:
            raise KeyError(f"handler of type {type} and name {name} not in {self.handlers}")

    def gather_and_store(self):
        for handler_type in self._data_handlers.keys():
            for handler_name in self._data_handlers[handler_type].keys():
                try:
                    self._data_handlers[handler_type][handler_name].gather_and_store()
                except Exception as e:
                    raise e

    def retrieve_data(self):
        data = {}
        for handler_type in self._data_handlers.keys():
            ht = {}
            for handler_name in self._data_handlers[handler_type].keys():
                try:
                    handler_data = self._data_handlers[handler_type][handler_name].retrieve()
                    ht[handler_name] = handler_data
                except Exception as e:
                    raise e
            data[handler_type] = ht

    @property
    def handlers(self):
        items = []
        for handler_type in self._data_handlers.keys():
            for handler_name in self._data_handlers[handler_type]:
                items.append((handler_type, handler_name))
        return items
