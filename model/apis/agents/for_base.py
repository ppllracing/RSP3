
from ..tools.config import Configuration
from ..tools.util import init_logger

class AgentBase:
    def __init__(self, cfg=None):
        self.cfg = Configuration() if cfg is None else cfg
        self.logger = init_logger(self.cfg, self.__class__.__name__)
        self.logger.info('Finish to Initialize Logger')

    def init_all(self):
        self.logger.info(f'Finish to Initialize Everything in {self.__class__.__name__}')