from functools import wraps
import time
import random
import pynvml
import platform
import psutil
import sys,os
import subprocess
import statistics
import shutil

def single(cls):
    _instance = {}
    @wraps(cls)
    def _single(*args,**kwargs):
        # nonlocal _instance
        if cls not in _instance :
            _instance[cls] = cls(*args,**kwargs) 
        else:
            pass
        return _instance[cls]
    return _single
def format_time() ->str:
    return time.strftime("%Y%m%d%H%M%S", time.localtime())
def has_multiple_keys(dictionary:dict, *keys)->bool:
    return set(keys).issubset(dictionary.keys())
def new_experiment_id() -> str:
        digits = [str(random.randint(0, 9)) for _ in range(15)]
        id = "".join(digits)
        return id
def get_gpu_list() ->list:
        device_list =[]
        try:
            pynvml.nvmlInit()
            device_count=pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_list.append(str(pynvml.nvmlDeviceGetName(handle)))
        except:
            #print("No Nvidia Gpu Or Bad Driver Version \n")
            pass
        return device_list
def get_os_info() ->dict:
        device = get_gpu_list()
        try:
            info = {
                    "hostname":platform.node(),
                    "platform":platform.platform(),
                    "system":platform.system(),
                    "python_version":platform.python_version(),
                    "architecture":platform.architecture()[0],
                    "processor":platform.processor(),
                    "uname":str(platform.uname()),
                    "cpu_logical_count":psutil.cpu_count(),
                    "cpu_count": psutil.cpu_count(logical=False),
                    "total_memory": psutil.virtual_memory().total /100000,
                    # "active_memory": psutil.virtual_memory().active /100000,
                    "available_memory": psutil.virtual_memory().available /100000,
                    # "total_swap_memory":psutil.swap_memory().total /100000,
                    "nvidia_gpu_info":str(device),
                    "python_path":sys.executable,
                    "run_path":os.getcwd()
            }
        except:
            raise BaseException("Failed To Collect Os Info")
        return info
def get_conda_info() ->str:
    output = ""
    try:
        result = subprocess.run(['conda', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
        output = result.stdout
        return output
    except:
        #print("Failed To Collect Conda Info")
        pass
    return output
def requirements_gen()->bool:
    return True if os.system("pip freeze > requirements.txt") == 0 else False
def quick_analysis(l:list) -> dict:
    def get_all_recorded_element(data)->list:
        elemet_list = []
        for d in data :
            elemet_list.extend(list(d.keys()))
        result = list(set(elemet_list))
        return result
    result = {}
    element_list = get_all_recorded_element(l)
    for e in element_list:
        result[e] = {}
        origin_list = []
        for s in l:
            if e in s.keys():
                origin_list.append(s[e])
        result[e]["max"] = max(origin_list)
        result[e]["min"] = min(origin_list)
        result[e]["viriance"]=statistics.variance(origin_list)
        result[e]["stdev"]=statistics.stdev(origin_list)
        result[e]["avg"] = statistics.mean(origin_list)
    return result
def all_analysis(d:dict)->dict:
    l = []
    for k in d.keys():
        l.extend(d[k])
    results = quick_analysis(l)
    return results
def copy_file_to_dir(srcfile,dstpath):                       
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       
        shutil.copy(srcfile, dstpath + fname)          