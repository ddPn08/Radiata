import logging
import sys

logger = logging.getLogger("radiata")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)
