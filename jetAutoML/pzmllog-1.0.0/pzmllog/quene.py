from multiprocessing import Queue as Q

#队列基类
class Quene():
    def __init__(self,len:int):
        self.__quene = Q(maxsize=len)
        return
    def Read(self):
        i = self.__quene.get()
        return i
    def Write(self,sth) ->None:
        i = self.__quene.put()
        return
    def Len(self) ->int:
        return self.__quene.qsize()