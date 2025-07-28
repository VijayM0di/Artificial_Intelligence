import datetime
from datetime import date, timedelta
import logging
import os.path
import time
# from settings import LOG_FILE_DAYS_BACKUP

import os
from dotenv import load_dotenv

load_dotenv()
LOG_FILE_DAYS_BACKUP = int(os.environ.get('LOG_FILE_DAYS_BACKUP'))


# Create a custom log file handler that rotates logs based on time
class TimedRotatingFileHandler(logging.FileHandler):
    def __init__(self, filename, when='h', interval=1, backupCount=1, encoding=None, delay=False, utc=False):
        self.when = when.upper()
        self.interval = interval
        self.backupCount = backupCount
        self.utc = utc
        self.suffix = filename

        if self.when == 'S':
            self.interval = 1  # One second
        elif self.when == 'M':
            self.interval = 60  # One minute
        elif self.when == 'H':
            self.interval = 3600  # One hour
        elif self.when == 'D':
            self.interval = 86400  # One day
        else:
            raise ValueError("Invalid 'when' parameter: Use S, M, H, or D")

        self.rolloverAt = self.computeRollover()
        self.backup_files = self.backupFiles()
        super().__init__(self.getFileName(), 'a', encoding, delay)


    def backupFiles(self):
        dt_now = datetime.datetime.now()
        dt_old = date.today() - timedelta(self.backupCount)
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        logfile_dir = ROOT_DIR.replace('helpers', 'logs')
        for f in os.listdir(logfile_dir):
            file = os.path.join(logfile_dir,f)
            file_date = f.split('_')[-1]
            if int(file_date.split('-')[0]) == dt_old.year:
                if int(file_date.split('-')[1]) == dt_old.month:
                    if int(file_date.split('-')[2].split('.')[0]) < dt_old.day:
                        os.remove(file)
                elif int(file_date.split('-')[1]) < dt_old.month:
                    os.remove(file)
            elif int(file_date.split('-')[0]) < dt_old.year:
                os.remove(file)




    def computeRollover(self):
        if self.utc:
            time_tuple = time.gmtime()
        else:
            time_tuple = time.localtime()
        if self.when == 'S':
            time_tuple = time.localtime()
        elif self.when == 'M':
            time_tuple = time.localtime()
            time_tuple = time_tuple[:5] + (0,) * 3
        elif self.when == 'H':
            time_tuple = time.localtime()
            time_tuple = time_tuple[:4] + (0,) * 2
        elif self.when == 'D':
            time_tuple = time.localtime()
            time_tuple = time_tuple[:3] + (0,) * 3 + (-1, -1, -1)
        current_time = int(time.mktime(time_tuple))
        remainder = current_time % self.interval
        return current_time + self.interval - remainder

    def getFileName(self):
        return time.strftime(self.suffix, time.localtime(self.rolloverAt - self.interval))

    def shouldRollover(self, record):
        if int(time.time()) >= self.rolloverAt:
            return 1
        return 0

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.baseFilename = self.getFileName()
        self.mode = 'a'
        self.stream = self._open()
        current_time = int(time.mktime(time.localtime()))
        self.rolloverAt = self.computeRollover()


# Create a logger with the custom log file handler
logger = logging.getLogger('engine_logger')
logger.setLevel(logging.DEBUG)

file_handler = TimedRotatingFileHandler('logs/%Y-%m-%d.log', when='D', interval=1, backupCount=LOG_FILE_DAYS_BACKUP)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Create a logger with the custom log file handler
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
file_handler = TimedRotatingFileHandler('logs/error_%Y-%m-%d.log', when='D', interval=1, backupCount=LOG_FILE_DAYS_BACKUP)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
error_logger.addHandler(file_handler)
error_logger.addHandler(console_handler)
