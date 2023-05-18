from abc import ABCMeta, ABC, abstractmethod
import exchange_calendars as xcals
from symbols import Symbol
from events import OrderEvent, FillEvent
from queue import Queue
from data_handlers import DataHandler


class Exchange(ABC):
    def __init__(self, name):
        self._name = name
        self._symbols = {}
        self._order_queue: DataHandler = DataHandler()

    @property
    def name(self):
        return self._name

    @property
    def symbols(self):
        return self._symbols

    def add_symbols(self, symbol: Symbol) -> None:
        self._symbols[symbol.name] = symbol

    def put_orders(self, orders: [OrderEvent]) -> None:
        for order in orders:
            self._order_queue.store_data(order)

    def put_order(self, order: OrderEvent) -> None:
        self._order_queue.put(order)

    @abstractmethod
    def _fill_order(self, order) -> FillEvent:
        raise NotImplementedError("Must implement order filling logic")

    def fill_orders(self) -> [FillEvent]:
        fill_events = []
        while not self._order_queue.empty():
            order = self._order_queue.get()
            fill_events.append(self._fill_order(order))
        return fill_events
