import os
import math
import gzip
import paddle as pd
import paddle.v2 as paddle
        
def data_reader(file_dir):
    def data_readers():
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + "/" + fi, "r") as f:
                for line in f:
                    query, t1, t2, label = line.strip('\n').split('\t')
                    if query == '' or t1 == '' or t2 == '':
                        continue
                    query_data = []
                    t1_data = []
                    t2_data = []
                    for word in query.split():
                        try:
                            query_data.append(word_index[word])
                        except:
                            query_data.append(0)
                    for word in t1.split():
                        try:
                            t1_data.append(word_index[word])
                        except:
                            t1_data.append(0)
                    for word in t2.split():
                        try:
                            t2_data.append(word_index[word])
                        except:
                            t2_data.append(0)
                    datas = []    
                    for data in [query_data, t1_data, t2_data]:
                        if len(data) == 20:
                            datas.append(data)
                        elif len(data) < 20:
                            data.extend([0 for _ in range(20 - len(data))])
                            datas.append(data)
                        else:
                            datas.append(data[:20])
                    query_data, t1_data, t2_data = datas
                    yield query_data, t1_data, t2_data, int(label)
    return data_readers
    
def fc_net(dict_size, class_dim=2, emb_dim=256, hid_dim=128):
    
    # input layers
    query_data = paddle.layer.data("query_word",
                             paddle.data_type.integer_value_sequence(dict_size))
    t1_data = paddle.layer.data("t1_word",
                             paddle.data_type.integer_value_sequence(dict_size))
    t2_data = paddle.layer.data("t2_word",
                             paddle.data_type.integer_value_sequence(dict_size))
    
    label = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))

    # embedding query layer
    query_emb = paddle.layer.embedding(input=query_data, size=emb_dim)
    t1_emb = paddle.layer.embedding(input=t1_data, size=emb_dim)
    t2_emb = paddle.layer.embedding(input=t2_data, size=emb_dim)
    
    fc_layer_size = [32, 16]
    fc_layer_init_std = [1.0 / math.sqrt(s) for s in fc_layer_size]
    
    query_fc1 = paddle.layer.fc(
        name="query_fc1",
        input=query_data,
        size=fc_layer_size[0],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[0]))
    pd.trainer_config_helpers.evaluators.value_printer_evaluator(query_fc1)
    query_fc2 = paddle.layer.fc(
        name="query_fc2",
        input=query_fc1,
        size=fc_layer_size[1],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[1]))
    t1_fc1 = paddle.layer.fc(
        name="t1_fc1",
        input=t1_emb,
        size=fc_layer_size[0],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[0]))
    t1_fc2 = paddle.layer.fc(
        name="t1_fc2",
        input=t1_fc1,
        size=fc_layer_size[1],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[1]))
    t2_fc1 = paddle.layer.fc(
        name="t2_fc1",
        input=t2_emb,
        size=fc_layer_size[0],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[0]))
    t2_fc2 = paddle.layer.fc(
        name="t2_fc2",
        input=t2_fc1,
        size=fc_layer_size[1],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[1]))

    # output layer
    cos_1 = paddle.layer.cos_sim(a=query_fc2, b=t1_fc2)
    cos_2 = paddle.layer.cos_sim(a=query_fc2, b=t2_fc2)
    
    concat = paddle.layer.concat([cos_1, cos_2])
    softmax = paddle.layer.fc(name="softmax", input=concat, size=2, act=paddle.activation.Softmax())

    return softmax, [cos_1, cos_2], label
    
def train_dnn_model(num_pass):
    
    # load word dictionary
    print "loading dictionary"

    dict_size = len(word_index)
    class_dim = 2
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            data_reader(train_files), buf_size=1000),  
        batch_size=5000)
    test_reader = paddle.batch(
        data_reader(test_files), batch_size=5000)
    # fc_net, cnn_net, rnn_net
    softmax, outputs, label = fc_net(dict_size, class_dim)
    cost = paddle.layer.cross_entropy_cost(input=softmax, label=label)
    
    # create parameters
    parameters = paddle.parameters.create(cost)
    
    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
            learning_rate = 0.01,
            regularization=paddle.optimizer.L2Regularization(rate=0.001),
            model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # add auc evaluator
    paddle.evaluator.auc(input=outputs, label=label)

    # create trainer
    trainer = paddle.trainer.SGD(
            cost=cost,
            parameters=parameters,
            update_equation=adam_optimizer,
            is_local=True)

    # Define end batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Train epoch %d, Batch %d, Cost %f, %s" % (event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            print "Test epoch %d, %s" % (event.pass_id, result.metrics)
            if not os.path.exists("./params"):
                os.makedirs("./params")

            
            if os.path.isfile("./params/%s%s%s" % ("params_", event.pass_id, ".tar.gz")):
                os.remove("./params/%s%s%s" % ("params_", event.pass_id, ".tar.gz"))
            
            with gzip.open("./params/%s%s%s" % ("params_", event.pass_id, ".tar.gz"), "w") as f:
                parameters.to_tar(f)

    feeding = {"query_word":0, "t1_word":1, "t2_word":2, "label":3}
    print "start training"
    trainer.train(
            reader=train_reader,
            event_handler=event_handler,
            feeding=feeding,
            num_passes=num_passes)
    
    print("Training finished")


    
if __name__ == "__main__":


    train_files = "./train_data/"
    test_files = "./test_data/"

    word_index = {}

    with open("word_dict/search_query_title/sort_word_dict_10_9_idx_100w", "r") as f:
        for line in f:
            pair = line.strip("\n").split("\t")
            word = pair[0]
            idx = int(pair[1]) + 1
            word_index[word] = idx

    TRUTH = ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]
    global cluster_train
    
    cluster_train = os.getenv('PADDLE_CLUSTER_TRAIN', "False") in TRUTH
    
    if not cluster_train:
        paddle.init(use_gpu=False, trainer_count=int(os.getenv("PADDLE_TRAINER_COUNT", "1")))
    else:
        paddle.init(use_gpu=False, 
            trainer_count=int(os.getenv("PADDLE_TRAINER_COUNT", "1")),
            port=int(os.getenv("PADDLE_PORT", "7174")),
            ports_num=int(os.getenv("PADDLE_PORTS_NUM", "1")),
            num_gradient_servers=int(os.getenv("PADDLE_NUM_GRADIENT_SERVERS", "1")),
            trainer_id=int(os.getenv("PADDLE_TRAINER_ID", "0")),
            pservers=os.getenv("PADDLE_PSERVERS", "127.0.0.1"))
    num_passes = 5
    train_dnn_model(num_passes)
    
