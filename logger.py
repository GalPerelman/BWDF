import os
import logging


class Logger(object):
    def __init__(self, name, LOGGING_DIR):
        name = name.replace('.log', '')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        if not os.path.exists(LOGGING_DIR):
            os.makedirs(LOGGING_DIR, exist_ok=True)

        file_name = os.path.join(LOGGING_DIR, '%s.log' % name)
        handler = logging.FileHandler(file_name)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s'
                                      'p%(process)s {%(pathname)s:%(lineno)d} function %(funcName)s'
                                      '%(levelname)s - %(message)s',
                                      '%m-%d %H:%M:%S')

        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        self._logger = logger

    def get(self):
        return self._logger
