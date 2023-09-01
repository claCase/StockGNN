from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.Environment.abstract.orders import Order


class Contract(ABC):
    def __init__(self, type):
        self._type = type
        self._orders = []

    def add_order(self, order):
        self._orders.append(order)

    @property
    def orders(self):
        return self._orders

    @property
    def type(self):
        return self._type


class Stock(Contract):
    def __init__(self):
        super().__init__(type="STK")


class Option(Contract):
    def __init__(self, strike):
        super().__init__(type="OPT")
        self._strike = strike

    @property
    def strike(self):
        return self._strike


class Futures(Contract):
    def __init__(self):
        super().__init__(type="FUT")


class Forex(Contract):
    def __init__(self, curr1: str, curr2: str):
        super().__init__(type="FOX")
        self._currency1 = curr1
        self._currency2 = curr2

    @property
    def currency1(self) -> str:
        return self._currency1

    @property
    def currency2(self) -> str:
        return self._currency2

    @property
    def pairs(self) -> str:
        return self._currency1 + self._currency2
