import numpy as np
import pandas as pd
from abc import ABC, ABCMeta, abstractmethod
from events import OrderEvent, FillEvent
from exchanges import Exchange
from datetime import time, datetime, timedelta


class Simulator(ABC):
    def __init__(self, name, update_frequency: timedelta = timedelta(days=1)):
        self._name = name
        self._exchanges: {str: Exchange} = {}
        self._update_frequency = update_frequency
        self._strategy = None
        self._clock = None

    @abstractmethod
    def step(self):
        raise NotImplementedError("Implement Step Function")

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

    def get_latest_data(self):
        for exchange in self._exchanges:
            for symbol in exchange.symbols:
                symbol.aggregate_data()

    def simulate(self, start: datetime, end: datetime):
        self._clock = start
        while self._clock <= end:
            self.step()

    @property
    def exchanges(self):
        return [exchange.name for exchange in self._exchanges]
