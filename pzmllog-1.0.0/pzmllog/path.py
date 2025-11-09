from .utils import single,format_time
from .quene import Quene
from multiprocessing import Queue, Value ,Manager
import os
from .format import Formater


PM = {}
def GetPM():
    return PM
def NewStoragePath(sp:str) ->None:
    PM["storage_path"] =sp
    os.makedirs(name=PM["storage_path"],mode=0o777,exist_ok=True)
    # os.makedirs(name=PM["storage_path"]+"/code",mode=0o777,exist_ok=True)
    return 
def GetStoragePath() ->str:
    sp = PM["storage_path"]
    return sp
def NewRunPath():
    __count = 0
    sp = GetStoragePath()
    while True:
        nrp = f"{sp}/run-{__count}"
        os.makedirs(name=nrp,mode=0o777,exist_ok=True)
        PM["run_path"] = nrp
        PM["cpu_path"] = nrp+"/watcher/cpu"
        os.makedirs(name=nrp+"/watcher/cpu",mode=0o777,exist_ok=True)
        PM["gpu_path"] = nrp+"/watcher/gpu"
        os.makedirs(name=nrp+"/watcher/gpu",mode=0o777,exist_ok=True)
        os.makedirs(name=nrp+"/files",mode=0o777,exist_ok=True)
        # Formater.append_fo_file(path=nrp+"/start.tag",sth=Formater.format_time())
        msg = yield nrp
        if msg ==None:
            __count+=1
def GetRunPath()->str:
    rp = PM["run_path"]
    return rp
def NewCpuFilePath() ->str:
    __count = 0
    cp = GetCpuPath()
    __cpu_log_count =0 
    while True:
        __cpu_log_count +=1
        if __cpu_log_count >= 360:
            __count +=1
            __cpu_log_count =0
        if not cp == GetCpuPath():
            cp = GetCpuPath()
            __count =0
        cfp = f"{cp}/cpu-{__count}.log"
        if not os.path.exists(cp):
            os.makedirs(cp,mode=0o777,exist_ok=True)
        PM["cpu_file_path"] = cfp
        # with open(cfp,"w"):
        #     pass
        yield cfp
def GetCpuFilePath()->str:
    __gen = NewCpuFilePath()
    while True:
        next(__gen)
        cfp = PM["cpu_file_path"] 
        yield cfp
def GetCpuPath()->str:
    cp = PM["cpu_path"]
    return cp
def GetGpuPath()->str:
    gp = PM["gpu_path"]
    return gp
def NewGpuFilePath():
    __count = 0
    gp = GetGpuPath()
    __gpu_log_count =0 
    while True:
        __gpu_log_count +=1
        if __gpu_log_count >= 360:
            __count +=1
            __gpu_log_count =0
        if not gp == GetGpuPath():
            gp = GetGpuPath()
            __count =0
        gfp = f"{gp}/gpu-{__count}.log"
        if not os.path.exists(gp):
            os.makedirs(gp,mode=0o777,exist_ok=True)
        PM["gpu_file_path"] = gfp
        # with open(gfp,"w"):
        #     pass
        yield gfp
def GetGpuFilePath() ->str:
    __gen = NewGpuFilePath()
    while True:
        next(__gen)
        gfp = PM["gpu_file_path"] 
        yield gfp