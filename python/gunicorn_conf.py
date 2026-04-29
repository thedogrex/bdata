from multiprocessing import cpu_count
import os

import logging

from gunicorn.glogging import Logger


# bind = "127.0.0.1:8081"

bind = f'unix:/datas/bdata/python/gunicorn.sock'

# Workers
# workers = cpu_count() + 1
workers = 1
worker_class = 'uvicorn.workers.UvicornWorker'
timeout = 60000


class SnapshotFilteredLogger(Logger):
    def access(self, resp, req, environ, request_time):
        try:
            method = environ.get('REQUEST_METHOD')
            path = environ.get('RAW_URI') or environ.get('PATH_INFO')
            if method == 'GET' and path == '/admin/snapshot':
                return
        except Exception:
            # If anything goes wrong, fall back to default logging behavior.
            pass

        return super().access(resp, req, environ, request_time)

def create_text_file(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        # Try to open the file for exclusive creation
        with open(file_path, 'x') as file:
            print(f"File '{file_path}' created successfully.")
    except FileExistsError:
        pass


# Logging Options
loglevel = 'info'
accesslog = f'/datas/logs/access_log.txt'
errorlog = f'/datas/logs/error_log.txt'
capture_output = True

logger_class = SnapshotFilteredLogger


create_text_file(accesslog)
create_text_file(errorlog)
