import logging

fmt = "[%(levelname)s] %(message)s"
logging.basicConfig(format=fmt, level=logging.INFO)
logger = logging.getLogger("radiata")
