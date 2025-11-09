import sys
from .mylogger import Logger

def NewLogger(config:dict,info:dict)->Logger:
    if( sys.version_info>(3,11) ) or (sys.version_info <(3,6)):
        raise EnvironmentError("仅支持python版本 3.6 3.7 3.8 3.9 3.10")
    else:
        l = Logger(config)
        l.Start(info)
        return l
