from typing import Mapping
import numpy as np
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod
from src.Environment.abstract.orders import OrderEvent, FillEvent
from src.Environment.abstract.data_handlers import DataHandler
from src.Environment.abstract.events import Event
from src.Environment.abstract.exchanges import Exchange
from datetime import time, datetime, timedelta
from src.Environment.abstract.strategy import Strategy
from src.Environment.abstract.symbols import Symbol
import asyncio
from asyncio.queues import Queue


class Simulator(ABC):
    def __init__(self, name, update_frequency: timedelta = timedelta(days=1)):
        self._name = name
        self._exchanges: {str: Exchange} = {}
        self._update_frequency = update_frequency
        self._strategy: Strategy = None
        self._clock = None
        self.latest_data: Mapping[str: Symbol]
        self._event_queue: Queue[Event] = Queue()
        self._event_handler_mapping: Mapping[str:Mapping[str:]]

    @abstractmethod
    def step(self, order_events: [OrderEvent]):
        latest_data = {name: {} for name in self.exchanges}
        for exchange in self._exchanges:
            for symbol in exchange.symbols:
                latest_data[exchange.name][symbol] = symbol.aggregate_data()
        self.execute_orders(order_events)

    def execute_orders(self, order_events: [OrderEvent]) -> {str: [FillEvent]}:
        for order in order_events:
            exchange = order.exchange
            self._exchanges[exchange].put_order(order)
        filled_events = {}
        for exchange in self._exchanges:
            filled_events[exchange.name] = exchange.fill_orders()
        return filled_events

    def add_exchange(self, exchange: Exchange):
        self._exchanges[exchange.name] = exchange

    def get_exchange(self, name: str) -> Exchange:
        return self._exchanges[name]

    def get_latest_symbol_data(self):
        for exchange in self._exchanges:
            for symbol in exchange.symbols:
                symbol.aggregate_data()

    def simulate(self, start: datetime, end: datetime):
        self._clock = datetime.now() - start
        while self._clock <= end:
            self.step()
            delta = datetime.now() - self._clock

    @property
    def exchanges(self):
        return [exchange.name for exchange in self._exchanges]

    def set_strategy(self, strategy: Strategy):
        self._strategy = strategy

    def get_strategy(self):
        return self._strategy
