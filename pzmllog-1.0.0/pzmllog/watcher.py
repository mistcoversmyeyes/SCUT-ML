from multiprocessing import Process
from .path import NewCpuFilePath,GetCpuFilePath,NewGpuFilePath,GetGpuFilePath,PM
import time
import psutil
from .format import Formater
import pynvml
import threading

# def cpu_watch_loop():
#     _gen = cpu_info_gen()
#     _path_gen = GetCpuFilePath()
#     _path = next(_path_gen)
#     for _info in _gen:
#         # print('路径是:' + _path + ' ' + _info)
#         Formater.append_fo_file(str(_path),_info)
#         continue

def cpu_watch_loop():
    _gen = cpu_info_gen()
    _path_gen = GetCpuFilePath()
    for _info in _gen:
        _path = next(_path_gen)
        Formater.append_fo_file(_path, _info)
        continue


def cpu_info_gen():
        while True:
            data = {
                    "time":time.strftime('%Y-%m-%d %X', time.localtime()),
                    "cpu_percent":psutil.cpu_percent(),
                    "memory":psutil.virtual_memory().used
            }
            yield str(data)+"\n"
            time.sleep(5)
def NewCpuWatcher():
    t = threading.Thread(target=cpu_watch_loop,daemon=True)
    t.start()
    return t
def gpu_watch_loop():
        _gen = gpu_info_gen()
        _path_gen = GetGpuFilePath()
        for _info in _gen:
            _path = next(_path_gen)
            Formater.append_fo_file(str(_path),_info)
            continue
def gpu_info_gen():
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            while True:
                device_status =[]
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    status = {"time":Formater.format_time(),"gpu_percent":gpu_percent.gpu,"gpu_memory":gpu_memory.used}
                    device_status.append(status)
                yield str(device_status)+"\n"
                time.sleep(5)
        except:
            #print("No Nvidia GPU Or Bad Driver Version")
            yield ""
def NewGpuWatcher():
    t = threading.Thread(target=gpu_watch_loop,daemon=True)
    t.start()
    return t



