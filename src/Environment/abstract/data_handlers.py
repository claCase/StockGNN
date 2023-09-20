from typing import List, Generic
from abc import ABC, abstractmethod
from src.Environment.abstract.events import (DataEvent,
                                             GatherEvent,
                                             MaximumTimeExceeded,
                                             DataProcessEvent,
                                             DataStoreEvent)
from queue import Queue, PriorityQueue
from typing import Mapping
from datetime import datetime, timedelta
import asyncio
from asyncio.queues import Queue
import pandas as pd


class Consumer(ABC):
    """
    Abstract Class used to define the logic for processing incoming events and store data. The architecture is based on
    asynchronous producer-consumer logic. This class is the consumer, the Producer class is the producer.
    The Producer class posts data onto the main event queue, where events are dispatched to the respective data handlers.
    The Consumer class processes and stores the data from the internal event queue.
    """

    def __init__(self,
                 type,
                 name="default",
                 data_wait_time: timedelta = timedelta(seconds=2),
                 max_process_time: timedelta = timedelta(seconds=500),
                 max_storing_time=timedelta(seconds=500)):
        self._type = type
        self._name = name
        self._data_wait_time = data_wait_time
        self._max_process_time = max_process_time
        self._max_storing_time = max_storing_time
        self._data_queue = PriorityQueue()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    async def put_data(self, data):
        await self._data_queue.put((data.datetime, data))

    @abstractmethod
    async def _preprocess(self, data: DataEvent) -> DataProcessEvent:
        """
            Pre-process data before storing it
        """
        raise NotImplementedError("Need to implement the pre-process  logic before storing the data")

    @abstractmethod
    async def _store_data(self, data_event: DataEvent) -> DataStoreEvent:
        """
            Stores the latest data items in Local Files or DataBase, it should return a DataEvent
        """
        raise NotImplementedError("Need to implement the data storing logic")

    async def run(self, event_queue: PriorityQueue):
        while True:
            try:
                priority, data = await self._data_queue.get()
                try:
                    processed_event = await asyncio.wait_for(self._preprocess(data),
                                                             self._max_process_time.total_seconds())
                    await event_queue.put((datetime.now(), processed_event))
                except asyncio.TimeoutError as time_error:
                    print(f"Data Processing for {self._type} {self._name} took longer than {self._max_process_time}")
                    await event_queue.put(MaximumTimeExceeded(self._type + "_" + self._name, datetime.now()))
                    raise time_error
                try:
                    store_event = await asyncio.wait_for(self._store_data(processed_event),
                                                         self._max_storing_time.total_seconds())
                    await event_queue.put((datetime.now(), store_event))
                except asyncio.TimeoutError as time_error:
                    print(f"Data Saving for {self._type} {self._name} took longer than {self._max_storing_time}")
                    await event_queue.put(MaximumTimeExceeded(self._type + "_" + self._name, datetime.now()))
                    raise time_error
            except asyncio.QueueEmpty as eq:
                print(f"DataStore {self._type} {self._name} has not received data, waiting...")
                await asyncio.sleep(self._data_wait_time.seconds)
                continue
            except KeyboardInterrupt as ek:
                print("Keyboard Interrupt by user, exiting main loop")
                break
            except Exception as e:
                raise e


class Producer(ABC):
    def __init__(self, type, name="default",
                 max_gathering_time: timedelta = timedelta(days=1000)):
        self._type = type
        self._name = name
        self._latest_time = datetime.now()
        self._max_gathering_time = max_gathering_time
        self._is_done = False

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @abstractmethod
    async def _gather_data(self) -> GatherEvent:
        """
        Gathers latest data from either APIs, DataBases or Local Files, flag self.is_done = True for stopping the main
        run loop. It returns a gather event.
        """
        raise NotImplementedError

    async def run(self, event_queue: PriorityQueue):
        while not self._is_done:
            try:
                try:
                    data = await asyncio.wait_for(self._gather_data(), self._max_gathering_time.total_seconds())
                    await event_queue.put((datetime.now(), data))
                except asyncio.TimeoutError as time_error:
                    await event_queue.put((datetime.now(), MaximumTimeExceeded(self._type + "_" + self._name,
                                                                               datetime.now())))
                    raise time_error
            except KeyboardInterrupt as ek:
                print("Keyboard Interrupt by user, exiting main loop")
                break


class DataStorer(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def store_data(self, data):
        raise NotImplementedError


class DataLoader(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def load_data(self):
        raise NotImplementedError


class GatherStore(ABC):
    def __init__(self):
        self._data_handlers: Mapping[str:Mapping[str:Consumer]] = {}

    def add_data_handler(self, data_handler: Consumer):
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
