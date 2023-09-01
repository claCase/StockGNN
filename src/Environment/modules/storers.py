import os
from src.Environment.abstract.data_handlers import DataHandler
from src.Environment.abstract.events import DATA_PROCESS_MESSAGES, DATA_STORE_MESSAGES, DATA_GATHER_MESSAGES, \
    GatherEvent
from src.Environment.modules.events import CSVEventSave, CSVEventProcess, CSVEventGather
from datetime import datetime, timedelta
from src.Data.data import df_to_matrix, TimeSeriesBatchGenerator, StockTimeSeries
import numpy as np
import pandas as pd
from asyncio import Queue


# from src.Data.data import


class DataFrameHandler(DataHandler):
    """
    Abstract Class used to define the logic for processing incoming events and store data. The architecture is based on
    asynchronous producer-consumer logic. This class is the consumer, the DataGather class is the producer.
    The DataGather class posts data onto the main event queue, where events are dispatched to the respective data handlers.
    The DataHandler class processes and stores the data from the internal event queue.
    """

    def __init__(self,
                 save_path: os.path,
                 name="default",
                 data_wait_time=2,
                 max_process_time: timedelta = timedelta(seconds=500),
                 max_storing_time=timedelta(seconds=500)):
        super(DataFrameHandler, self).__init__("CSV", name, data_wait_time, max_process_time, max_storing_time)
        self._save_path = save_path

    async def _preprocess(self, data_event):
        data = data_event.data
        return CSVEventProcess(data=data, msg=DATA_PROCESS_MESSAGES("PROCESSED"), datetime=datetime.now())

    async def _store_data(self, data_event: CSVEventProcess):
        data = data_event.data
        try:
            full_path = os.path.join(self._save_path, data_event.datetime)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            data.to_csv(os.path.join(full_path, self.name))
            return CSVEventSave(data, msg=DATA_STORE_MESSAGES("SAVED"), datetime=datetime.now())
        except IOError as e:
            return CSVEventSave(data, msg=DATA_STORE_MESSAGES("SAVE_FAILED"), datetime=datetime.now())
        except Exception as e:
            raise e


class StockTimeSeriesHandler(DataHandler):
    def __init__(self,
                 save_path: os.path,
                 name="default",
                 exchange='',
                 data_wait_time=2,
                 max_process_time: timedelta = timedelta(seconds=500),
                 max_storing_time=timedelta(seconds=500)):
        super(StockTimeSeriesHandler, self).__init__("STOCK_SERIES", name, data_wait_time, max_process_time,
                                                     max_storing_time)
        self._save_path = save_path
        self._exchange = exchange

    async def _preprocess(self, data):
        assert isinstance(data, CSVEventGather) and data.message == DATA_GATHER_MESSAGES.GATHERED
        data = data.data
        data = df_to_matrix(data)

    async def _store_data(self, data_event):
        pass
