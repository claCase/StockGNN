import os
from datetime import datetime, timedelta
from ..abstract.data_handlers import Storer
from ..abstract.events import DATA_PROCESS_MESSAGES, DATA_STORE_MESSAGES, DATA_GATHER_MESSAGES
from ..modules.events import CSVEventSave, CSVEventProcess, CSVEventGather, StockTimeseriesDataFrame


# from src.Data.data import


class CSVStorerFlatFile(Storer):
    """
    Abstract Class used to define the logic for processing incoming events and store data. The architecture is based on
    asynchronous producer-consumer logic. This class is the consumer, the Producer class is the producer.
    The Producer class posts data onto the main event queue, where events are dispatched to the respective data handlers.
    The Consumer class processes and stores the data from the internal event queue.
    """

    def __init__(self,
                 save_path: os.path,
                 name="default",
                 data_wait_time=2,
                 max_storing_time=timedelta(seconds=500)):
        super().__init__(max_storing_time, "CSVStorerFlatFile", name, data_wait_time)
        self._save_path = save_path

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