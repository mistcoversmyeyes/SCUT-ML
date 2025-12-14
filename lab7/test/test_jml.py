from pzmllog import NewLogger

data_length = 10
epoch = 3
learning_rate = 0.001

log = NewLogger(
    config={
        #用户 ID
        'access_token':"your token",
        #项目 ID
        'project':"1392",
        #实验描述和说明信息
        "description":"test",
        #自定义实验名称
        "experiment_name":"TEST",
        #仓库 ID
        "repository_id":"your rep id",
        # tomcat的启动端口
        'port': "5560"
    },
    #超参数集
    info = {
        "learning_rate": learning_rate,
        "epoch": epoch,
        "batch_size": 64
    }
)


for e in range(epoch):
    # 开始实验
    log.Run()
    for i in range(data_length):
        # 实验部分
        loss = 0.1*(10 - i)
        accuracy = (0.1 * i)
        # 记录 Log
        log.Log({"epoch":epoch,"loss":loss,"accuracy":accuracy})
    # 实验模型路径
    model_path = "model.pt"
    # 记录模型
    log.Save([model_path])
    # 结束实验
    log.End()
# 结束整个过程
log.Submit()