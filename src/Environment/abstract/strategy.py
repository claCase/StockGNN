from abc import ABCMeta, ABC, abstractmethod
import exchange_calendars as xcals
from symbols import Symbol
from src.Environment.abstract.orders import OrderEvent
from queue import Queue
from data_handlers import DataHandler, GatherStore


class Strategy(ABC, GatherStore):
    def __init__(self, name=""):
        self._data_handlers: {{DataHandler}} = {}
        self._name = name
        self._orders = Queue()

    @abstractmethod
    def create_order(self):
        raise NotImplementedError

    def add_order(self, order: OrderEvent):
        self._orders.put(order)

    def get_orders(self):
        return self._orders

    @property
    def name(self):
        return self._name

    @abstractmethod
    def step(self):
        raise NotImplementedError
