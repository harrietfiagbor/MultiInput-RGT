import boto3
import json
import time
import os
import logging
from optuna_utils import run_study

# LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
# logging.basicConfig(level=LOGLEVEL)
# logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)

logging.debug("test log debug message")
logging.info("test log info message")
logging.warning("test log warning message")
logging.error("test log error message")

logging.debug(os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"))

logging.info("Importing tsai...")
import tsai

logging.info("tsai imported.")

# bucket = boto3.resource('s3').Bucket('ml-gpu-example')
logging.info("Connecting to queue...")
queue = boto3.resource("sqs").get_queue_by_name(QueueName="Semler-Queue")
logging.info("Connected to queue.")


def startWorker():
    while True:
        logging.info("Polling for messages...")
        max_messages = 1
        logging.info(f"max_messages = {max_messages}")
        for message in queue.receive_messages(MaxNumberOfMessages=max_messages):
            config = json.loads(message.body)
            logging.debug(f"config = {config}")
            logging.info("Running study...")
            run_study(config)
            logging.info("Study done.")
            logging.info("Deleting message from queue...")
            message.delete()
            logging.info("Message deleted from queue.")
        sleep_secs = 30
        logging.info(f"Sleeping for {sleep_secs} seconds...")
        time.sleep(sleep_secs)


startWorker()
