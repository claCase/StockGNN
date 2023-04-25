from abc import ABC, abstractmethod
from events import Event


class DataHandler(ABC):
    def __init__(self, type):
        self._type = type

    @property
    def type(self):
        return self._type

    @abstractmethod
    def gather_data(self) -> [Event]:
        raise NotImplementedError("Implement Logic for Gathering data events")
