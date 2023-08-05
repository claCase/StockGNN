from abc import ABC, abstractmethod
from events import Event
from queue import Queue
from typing import Mapping
from datetime import datetime
import asyncio


class DataEvent(Event):
    def __init__(self, type, msg, datetime: datetime):
        super().__init__(type=type, datetime=datetime)
        self._msg = msg

    @property
    def message(self):
        return self._msg

    @abstractmethod
    def check_message_code(self):
        raise NotImplementedError("Implement message checking logic based on type of DataEvent")


class DataStoreEvent(DataEvent):
    def __init__(self, msg, datetime: datetime):
        super().__init__(type="DS", msg=msg, datetime=datetime)
        self.DATA_STORE_MESSAGES = {}

    def check_message_code(self):
        assert self.message in self.DATA_STORE_MESSAGES


class DataGatherEvent(DataEvent):
    def __init__(self, msg, datetime: datetime):
        super().__init__(type="DG", msg=msg, datetime=datetime)
        self.DATA_GATHER_MESSAGES = {}

    def check_message_code(self):
        assert self.message in self.DATA_GATHER_MESSAGES


class DataHandler(ABC):
    """
    Abstract Class used to define the logic for processing incoming events and store data. The architecture is based on
    asynchronous producer-consumer logic. This class is the consumer, the DataGather class is the producer.
    The DataGather class posts data onto the main event queue and the DataHandler class processes and stores the data
    """

    def __init__(self, type, name="default"):
        self._type = type
        self._name = name

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @abstractmethod
    async def _preprocess(self, data):
        """
            Preprocess data before storing it
            """
        raise NotImplementedError("Need to implement the preprocess  logic before storing the data")

    async def preprocess(self, data):
        await self._preprocess(data)

    @abstractmethod
    async def _store_data(self, data) -> DataStoreEvent:
        """
            Stores the latest data items in Local Files or DataBase, it should return a DataStoreEvent
            """
        raise NotImplementedError("Need to implement the data storing logic")

    async def store_data(self, data):
        await self._store_data(data)


class DataGather(ABC):
    def __init__(self, type, name="default"):
        self._type = type
        self._name = name

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @abstractmethod
    async def _gather_data(self) -> ([Event], DataGatherEvent):
        """
        Gathers latest data from either APIs, DataBases or Local Files
        """
        raise NotImplementedError

    async def gather_data(self):
        return await self._gather_data()


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
