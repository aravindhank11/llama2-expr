from collections import deque
import threading
import time
from enum import Enum
import struct
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, KafkaTimeoutError
from kafka import KafkaProducer, KafkaConsumer


def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)


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


class KafkaClient:
    def __init__(self, kafka_server, topic, topic_creation_mode):
        self._kafka_server = kafka_server
        self._topic = str(topic)
        if topic_creation_mode == TopicCreationMode.FORCE_NEW_CREATE:
            self._force_create_new_topic()
        elif topic_creation_mode == TopicCreationMode.WAIT_TO_CREATE:
            self._wait_till_topic_create()
        elif topic_creation_mode == TopicCreationMode.CREATE_IF_NO_EXIST:
            self._create_topic_if_not_exist()

    def _force_create_new_topic(self):
        # Create an admin client
        admin_client = KafkaAdminClient(bootstrap_servers=self._kafka_server)

        # If topic exists => delete it
        if self._topic in admin_client.list_topics():
            admin_client.delete_topics([self._topic])

        # Create the topic again
        topic_to_create = NewTopic(
            name=self._topic, num_partitions=1, replication_factor=1
        )

        # Close the admin client
        admin_client.create_topics(new_topics=[topic_to_create])

    def _wait_till_topic_create(self):
        # Create an admin client
        admin_client = KafkaAdminClient(bootstrap_servers=self._kafka_server)

        # Waiting for topic to be created
        while self._topic not in admin_client.list_topics():
            time.sleep(0.25)

        # Close the admin client
        admin_client.close()

    def _create_topic_if_not_exist(self):
        # Create an admin client
        admin_client = KafkaAdminClient(bootstrap_servers=self._kafka_server)

        # Check if a topic exist
        if self._topic not in admin_client.list_topics():
            try:
                # Create the topic
                topic_to_create = NewTopic(
                    name=self._topic, num_partitions=1, replication_factor=1
                )
                admin_client.create_topics(new_topics=[topic_to_create])
            except TopicAlreadyExistsError:
                pass

            # Close the admin client
            admin_client.close()

    def dump_queue(self, python_queue):
        producer = KafkaProducer(bootstrap_servers=self._kafka_server)
        successful_sends = 0
        while not python_queue.empty():
            item = python_queue.get()
            msg = struct.pack('f', item)
            producer.send(self._topic, msg)
            successful_sends += 1

        # -1 indicates end of produce
        msg = struct.pack('f', -1)
        producer.send(self._topic, msg)

        producer.flush()
        return successful_sends

    def read_queue(self, python_queue):
        consumer = KafkaConsumer(
            self._topic,
            bootstrap_servers=self._kafka_server,
            auto_offset_reset="earliest"
        )

        msgs_rcvd = 0
        for message in consumer:
            item = struct.unpack('f', message.value)[0]
            if item == -1:
                break
            python_queue.put(item)
            msgs_rcvd += 1

        consumer.close()
        return msgs_rcvd

    def dump_pkl(self, data):
        producer = KafkaProducer(bootstrap_servers=self._kafka_server)
        producer.send(self._topic, data)
        producer.flush()

    def read_pkl(self, timeout_ms=1000):
        consumer = KafkaConsumer(
            self._topic,
            bootstrap_servers=self._kafka_server,
            auto_offset_reset="earliest",
            consumer_timeout_ms=timeout_ms
        )

        messages = []
        try:
            for message in consumer:
                messages.append(message.value)
        except KafkaTimeoutError:
            pass

        consumer.close()
        return messages


class DistributionType(Enum):
    CLOSED = (1, "CLOSED")
    POINT = (2, "POINT")
    POISSON = (3, "POISSON")


class WatermarkType(Enum):
    HIGH = (1, "HIGH")
    LOW = (2, "LOW")


class TopicCreationMode(Enum):
    FORCE_NEW_CREATE = (1, "FORCE_NEW")
    WAIT_TO_CREATE = (2, "WAIT_TO_CREATE")
    CREATE_IF_NO_EXIST = (3, "CREATE_IF_NO_EXIST")
