from .path import GetRunPath
from .format import Formater
from .utils import quick_analysis,all_analysis,single
@single
class DataStatusManager():
    def __init__(self,config:dict) ->None:
        self.__config = config
        self.__status = {}
        return
    def Log(self,statu:dict) ->None:
        rp = GetRunPath()
        if not rp in self.__status.keys():
            self.__status[rp] = []
        self.__status[rp].append(statu)
        return
    def WriteToDisk(self,rp:str) ->None:
        Formater.save_dict_to_json(dict_value=self.__status[rp],save_path=rp+"/results.json")
        Formater.save_list_dict_to_csv(list_dict=self.__status[rp],output_file=rp+"/results.csv")
        run_analysis = quick_analysis(l=self.__status[rp])
        Formater.save_dict_to_json(dict_value=run_analysis,save_path=rp+"/analysis.json")
        Formater.save_dict_to_yaml(dict_value=run_analysis,save_path=rp+"/analysis.yaml")
        Formater.save_dict_to_json(dict_value=self.__status[rp][-1],save_path=rp+"/last.json")
        Formater.save_dict_to_yaml(dict_value=self.__status[rp][-1],save_path=rp+"/last.yaml")
        return
    def AllAnalysis(self,sp)->None:
        all = all_analysis(d= self.__status)
        Formater.save_dict_to_json(dict_value=all,save_path=sp+"/analysis.json")
        Formater.save_dict_to_yaml(dict_value=all,save_path=sp+"/analysis.yaml")
def NewStatus(config:dict) ->DataStatusManager:
    return DataStatusManager(config=config)