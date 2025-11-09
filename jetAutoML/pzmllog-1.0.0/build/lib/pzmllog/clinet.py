import requests
import os
from .path import NewStoragePath, GetStoragePath, NewRunPath, GetRunPath


class Clinet(object):
    def __init__(self, config: dict, host="127.0.0.1") -> None:
        port = config['port']
        self.__api_load_save_path = f"http://{host}:{port}/ml_client/client/loadSavePath"
        self.__api_notice_experiment = f"http://{host}:{port}/ml_client/client/noticeExperiment"
        self.__api_notice_run = f"http://{host}:{port}/ml_client/client/noticeRun"
        self.__config = config
        return

    def __api(self, url: str, data: dict) -> dict:
        try:
            header = {'Content-Type': 'application/json'}
            resp = requests.post(url=url, headers=header, json=data)
            # print(f"Send Message:{data}\n")
            msg = dict(resp.json())
            # print(f"Client Response: {msg}\n")
        except:
            raise ConnectionError("Client Connection Error,Check Client's Status Please\n")
        if not msg["code"] == 200:
            raise ConnectionError("Client Start Successful, But Has Internal Error\n")
        return msg

    def ShakeHand(self) -> None:
        send_data = {}
        send_data["userToken"] = self.__config["access_token"]
        send_data["projectId"] = self.__config["project"]
        send_data["description"] = self.__config["description"]
        send_data["experimentName"] = self.__config["experiment_name"]
        if "repository_id" in self.__config:
            send_data["repositoryId"] = self.__config["repository_id"]
        resp = self.__api(url=self.__api_load_save_path, data=send_data)
        NewStoragePath(resp["data"] + "/" + self.__config["experiment_id"])
        return

    def NoticeExpStart(self) -> None:
        send_data = {
            "experimentId": self.__config["experiment_id"],
            "status": 0
        }
        self.__api(url=self.__api_notice_experiment, data=send_data)
        return

    def NoticeRunStart(self) -> None:
        run_path = os.path.basename(GetRunPath())
        send_data = {
            "experimentId": self.__config["experiment_id"],
            "runName": run_path,
            "status": 0
        }
        self.__api(url=self.__api_notice_run, data=send_data)
        return

    def NoticeRunStop(self) -> None:
        run_path = os.path.basename(GetRunPath())
        send_data = {
            "experimentId": self.__config["experiment_id"],
            "runName": run_path,
            "status": 1
        }
        self.__api(url=self.__api_notice_run, data=send_data)
        return

    def NoticeExpStop(self) -> None:
        send_data = {
            "experimentId": self.__config["experiment_id"],
            "status": 1
        }
        self.__api(url=self.__api_notice_experiment, data=send_data)
        return


def NewClientConn(config: dict) -> Clinet:
    return Clinet(config=config)
