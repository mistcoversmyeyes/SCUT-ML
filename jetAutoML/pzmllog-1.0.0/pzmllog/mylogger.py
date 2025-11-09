
from .status import NewStatus
from .clinet import NewClientConn
from .path import GetStoragePath,GetRunPath,GetPM,NewRunPath
from .utils import copy_file_to_dir,get_os_info,single,format_time,has_multiple_keys,new_experiment_id,get_conda_info,requirements_gen
from .watcher import NewCpuWatcher,NewGpuWatcher
from .format import Formater
import shutil
import os

@single
class Logger(object):
    def __init__(self,config:dict)->None:
        if not has_multiple_keys(config, 'access_token', 'project',"description","experiment_name"):
            raise BaseException("Params Missing: access_token project description experiment (repository_id)")
        config["experiment_id"] = new_experiment_id()
        self.__config = config
        self.__config["experiment_name"] = f'{self.__config["experiment_name"]}-{format_time()}'
        self.__status = NewStatus(config=self.__config)
        self.__client = NewClientConn(config=self.__config)
        self.__client.ShakeHand()
        self.__rp_gen = None
        self.__cpu =None
        self.__gpu =None
        self.__rlock = 0
        return
    def Start(self,info:dict)->None:
        self.__super_arg = info
        self.__save_code()
        self.__client.NoticeExpStart()
        self.__start_tag()
        self.__save_config()
        self.__conda_info()
        self.__save_requirements()
        self.__os_info()
        self.__save_super_arg()

        return
    def Run(self)->None:
        # print(GetPM())
        if self.__rp_gen == None:
            self.__rp_gen = NewRunPath()
        next(self.__rp_gen)
        self.__run_start_tag()
        self.__client.NoticeRunStart()
        self.__cpu_watcher() if self.__cpu == None else ()
        self.__gpu_watcher() if self.__gpu == None else ()
        return
    def Log(self,statu:dict)->None:
        self.__status.Log(statu=statu)
        return
    def End(self)->None:
        rp = GetRunPath()
        self.__status.WriteToDisk(rp=rp)
        self.__client.NoticeRunStop()
        self.__run_finish_tag()
        pass
    def Submit(self)->None:
        self.__all_analysis()
        self.__finish_tag()
        self.__client.NoticeExpStop()
        return
    def Print(self,words:str)->None:
        print("[Logger.Print()] ->",words)
        Formater.append_fo_file(path=GetRunPath()+"/console.log",sth=str(words)+"\n")
    def Save(self,file_path_list:list)->None:
        for path in file_path_list:
            if os.path.exists(path):
                file_name = os.path.basename(path)
                print(f"[Logger.Save()] ->Save File : {file_name}")
                copy_file_to_dir(srcfile=file_name,dstpath=f"{GetRunPath()}/files/")
                Formater.append_fo_file(path=GetRunPath()+"/file.tag",sth=f"files/{file_name}\n")
            else:
                print(f"[Logger.Save()] ->No Such A File: {path} ,Check Your Project Current Location \n")
        return
    def __save_code(self,path=["datasets"])->None:
        ignore_path = [*path]
        try:
            if os.path.exists(".path_ignore"):
                with open(".path_ignore","r") as f:
                    line =  f.readline()
                    while line:
                        ignore_path.append(line.strip())
                        line = f.readline()
            dst_dir =GetStoragePath()+"/code"
            shutil.copytree(src=os.getcwd(),dst=dst_dir,ignore=shutil.ignore_patterns(*ignore_path))
        except:
            print("Source Code Backup Failed \n")
        return
    def __save_config(self) ->None:
        Formater.save_dict_to_json(dict_value=self.__config,save_path=GetStoragePath()+"/config.json")
        Formater.save_dict_to_yaml(dict_value=self.__config,save_path=GetStoragePath()+"/config.yaml")
        return   
    def __save_requirements(self)->None:
        (shutil.copy("./requirements.txt",f"{GetStoragePath()}")) if requirements_gen() else ()
        return
    def __conda_info(self) ->None:
        Formater.append_fo_file(path=GetStoragePath()+"/conda.info",sth=get_conda_info())
        return
    def __finish_tag(self)->None:
        path = GetStoragePath()+"/finish.tag"
        Formater.append_fo_file(path=path,sth=Formater.format_time())
        return
    def __start_tag(self)->None:
        path = GetStoragePath()+"/start.tag"
        Formater.append_fo_file(path=path,sth=Formater.format_time())
        return
    def __run_start_tag(self)->None:
        path = GetRunPath()+"/start.tag"
        Formater.append_fo_file(path=path,sth=Formater.format_time())
        return
    def __run_finish_tag(self)->None:
        path = GetRunPath()+"/finish.tag"
        Formater.append_fo_file(path=path,sth=Formater.format_time())
        return
    def __os_info(self) ->None:
        json_path = GetStoragePath()+"/os_info.json"
        yaml_path = GetStoragePath()+"/os_info.yaml"
        Formater.save_dict_to_json(dict_value=get_os_info(),save_path=json_path)
        Formater.save_dict_to_yaml(dict_value=get_os_info(),save_path=yaml_path)
        return
    def __save_super_arg(self) ->None:
        json_path = GetStoragePath()+"/super_arg.json"
        yaml_path = GetStoragePath()+"/super_arg.yaml"
        Formater.save_dict_to_json(dict_value=self.__super_arg,save_path=json_path)
        Formater.save_dict_to_yaml(dict_value=self.__super_arg,save_path=yaml_path)
        return
    def __all_analysis(self) ->None:
        self.__status.AllAnalysis(sp=GetStoragePath())
        return
    def __cpu_watcher(self) ->None:
        self.__cpu =NewCpuWatcher()
        return
    def __gpu_watcher(self)->None:
        self.__gpu =NewGpuWatcher()
        return
    

