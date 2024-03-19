from collections import deque
import threading
import time
from enum import Enum
import struct
import sysv_ipc
import psutil


def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)


def is_process_alive(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except psutil.NoSuchProcess:
        return False


def orion_block(backend_lib, it):
    # block client until request served
    backend_lib.block(it)


class OnlinePercentile:
    def __init__(self, p, K):
        self.p = p
        self.K = K
        self.data_stream = deque(maxlen=K)

    def add_element(self, value):
        self.data_stream.append(value)

    def get_pth_percentile(self):
        if not self.data_stream:
            return -1

        # Sort the data stream
        sorted_stream = sorted(self.data_stream)

        # Calculate the index for the p-th percentile
        index = int(self.p * len(sorted_stream) / 100)

        # If the index is not an integer, interpolate the value
        if index != len(sorted_stream) * self.p / 100:
            lower_bound = sorted_stream[index - 1]
            upper_bound = sorted_stream[index]
            return (lower_bound + upper_bound) / 2.0
        else:
            return sorted_stream[index - 1]


class AtomicBoolean:
    def __init__(self, initial_value=False):
        self._lock = threading.Lock()
        self._value = initial_value

    def flip(self):
        with self._lock:
            self._value = not self._value

    def get(self):
        with self._lock:
            return self._value


class SysVQueue:
    def __init__(self, qid, create_new_queue=True):
        self._qid = qid
        if create_new_queue:
            self._force_create_new_queue()
        else:
            self._wait_till_queue_create()

    def _force_create_new_queue(self):
        try:
            self._sysvq = sysv_ipc.MessageQueue(self._qid)
            self._sysvq.remove()
        except sysv_ipc.ExistentialError:
            pass
        finally:
            self._sysvq = sysv_ipc.MessageQueue(
                self._qid, flags=sysv_ipc.IPC_CREX
            )

    def _wait_till_queue_create(self):
        while True:
            try:
                self._sysvq = sysv_ipc.MessageQueue(self._qid)
                break
            except sysv_ipc.ExistentialError:
                time.sleep(0.25)

    def dump_queue(self, python_queue):
        successful_sends = 0
        while not python_queue.empty():
            item = python_queue.get()
            msg = struct.pack('f', item)
            self._sysvq.send(msg)
            successful_sends += 1
        return successful_sends

    def read_queue(self, python_queue):
        msgs_rcvd = 0
        while True:
            try:
                message, _ = self._sysvq.receive(block=False)
                item = struct.unpack('f', message)[0]
                python_queue.put(item)
                msgs_rcvd += 1
            except sysv_ipc.BusyError:
                if not is_process_alive(self._qid):
                    self._sysvq.remove()
                    break
        return msgs_rcvd


class DistributionType(Enum):
    CLOSED = (1, "CLOSED")
    POINT = (2, "POINT")
    POISSON = (3, "POISSON")


class WatermarkType(Enum):
    HIGH = (1, "HIGH")
    LOW = (2, "LOW")
