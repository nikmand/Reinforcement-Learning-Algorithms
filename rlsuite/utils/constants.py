from enum import Enum

LOGGER = 'simpleExample'

logger_path = ""

class RLAlgorithms(Enum):
    Q_LEARNING = '0'
    SARSA = '1'
    DOUBLE_Q_LEARNING = '2'
