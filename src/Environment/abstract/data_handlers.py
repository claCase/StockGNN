from typing import List, Generic
from abc import ABC, abstractmethod
from src.Environment.abstract.events import Event, DataEvent, GatherEvent, MaximumTimeExceeded
from queue import Queue
from typing import Mapping
from datetime import datetime, timedelta
import asyncio
from asyncio.queues import Queue
import pandas as pd


class DataHandler(ABC):
    """
    Abstract Class used to define the logic for processing incoming events and store data. The architecture is based on
    asynchronous producer-consumer logic. This class is the consumer, the DataGather class is the producer.
    The DataGather class posts data onto the main event queue, where events are dispatched to the respective data handlers.
    The DataHandler class processes and stores the data from the internal event queue.
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
        self._data_queue = Queue()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    async def put_data(self, data):
        await self._data_queue.put(data)

    @abstractmethod
    async def _preprocess(self, data: DataEvent) -> DataEvent:
        """
            Pre-process data before storing it
        """
        raise NotImplementedError("Need to implement the pre-process  logic before storing the data")

    async def preprocess(self, data_event: DataEvent):
        processed = await self._preprocess(data_event)
        return processed

    @abstractmethod
    async def _store_data(self, data_event: DataEvent) -> DataEvent:
        """
            Stores the latest data items in Local Files or DataBase, it should return a DataEvent
        """
        raise NotImplementedError("Need to implement the data storing logic")

    async def store_data(self, data_event: DataEvent, queue: Queue):
        event = await self._store_data(data_event)
        await queue.put(event)

    async def run(self, queue: Queue):
        start_process = datetime.now()
        while True:
            try:
                data = await self._data_queue.get()
                current_time = datetime.now()
                if current_time <= start_process + self._max_process_time:
                    processed_event = await self.preprocess(data)
                else:
                    raise TimeoutError(
                        f"Data Processing for {self._type} {self._name} took longer than {self._max_process_time}")
                if current_time <= start_process + self._max_storing_time:
                    await self.store_data(processed_event, queue)
                else:
                    raise TimeoutError(
                        f"Data Storing for {self._type} {self._name} took longer than {self._max_process_time}")
            except asyncio.QueueEmpty as eq:
                print(f"DataStore {self._type} {self._name} has not received data, waiting...")
                await asyncio.sleep(self._data_wait_time.seconds)
                continue
            except KeyboardInterrupt as ek:
                print("Keyboard Interrupt by user, exiting main loop")
                break


class DataGather(ABC):
    def __init__(self, type, name="default", time_interval: timedelta = timedelta(seconds=1),
                 max_gathering_time: timedelta = timedelta(days=1000)):
        self._type = type
        self._name = name
        self._time_interval = time_interval
        self._latest_time = datetime.now()
        self._max_gathering_time = max_gathering_time
        self.is_done = False

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

    async def gather_data(self) -> GatherEvent:
        return await self._gather_data()

    async def run(self, queue: Queue):
        start_process = datetime.now()
        while True and not self.is_done:
            try:
                current_time = datetime.now()
                if current_time <= start_process + self._max_gathering_time:
                    try:
                        data = await self.gather_data()
                        await queue.put(data)
                    except Exception as e:
                        raise e
                else:
                    await queue.put(MaximumTimeExceeded(name=self._type + self._name, datetime=datetime.now()))
                    break
            except asyncio.QueueEmpty as eq:
                print(f"DataGather {self._type} {self._name} did not gather any data, waiting...")
                await asyncio.sleep(self._time_interval.seconds)
            except KeyboardInterrupt as ek:
                print("Keyboard Interrupt by user, exiting main loop")
                break


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
