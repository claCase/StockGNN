import os
from src.Environment.abstract.data_handlers import Consumer, Storer, Processor
from src.Environment.abstract.events import DATA_PROCESS_MESSAGES, DATA_STORE_MESSAGES, DATA_GATHER_MESSAGES
from src.Environment.modules.events import CSVEventSave, CSVEventProcess, CSVEventGather, StockTimeseriesDataFrame
from datetime import datetime, timedelta
from src.Modelling.data import StockTimeSeries


class StockTimeSeriesProcessor(Processor):
    def __init__(self,
                 save_path: os.path,
                 name="default",
                 exchange='NYSE',
                 data_wait_time=20,
                 max_process_time: timedelta = timedelta(seconds=500)):
        super().__init__(max_process_time, "STOCK_SERIES", name, data_wait_time)
        self._save_path = save_path
        self._exchange = exchange

    async def _preprocess(self, data: StockTimeseriesDataFrame):
        assert isinstance(data, StockTimeseriesDataFrame) and data.message == DATA_GATHER_MESSAGES.GATHERED
        data = data.data
        try:
            return StockTimeSeries(data, self._exchange)
        except Exception as e:
            raise e