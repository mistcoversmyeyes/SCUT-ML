#格式化静态方法类
import time
import json
import csv
import yaml

# import yaml
class Formater():
    def __init__(self)->None:
        return
    @staticmethod
    def format_time() ->str:
        return time.strftime("%Y-%m-%d %X", time.localtime())
    @staticmethod
    def timestamp() ->str:
        return time.strftime("%Y%m%d%H%M%S", time.localtime())
    @staticmethod
    def append_fo_file(path:str,sth:str):
        with open(path,"a") as file:
                file.write(sth)
                file.flush()
        return    
    @staticmethod
    def save_dict_to_json(dict_value:dict , save_path:str) ->None:
        with open(save_path, 'w') as file:
            file.write(json.dumps(dict_value, indent=2))
            file.flush()
        return
    @staticmethod
    def save_list_dict_to_csv(list_dict:list, output_file:str)->None:
        headers = set()
        for item in list_dict:
            headers.update(item.keys())
        with open(output_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            for item in list_dict:
                row = [item.get(key, '') for key in headers]
                writer.writerow(row)
        return
    @staticmethod
    def save_dict_to_yaml(dict_value: dict, save_path: str):
        with open(save_path, 'w') as file:
            file.write(yaml.dump(dict_value, allow_unicode=True))
            file.flush()
        return
