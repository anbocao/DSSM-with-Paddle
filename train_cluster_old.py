import os
import math
import gzip
import paddle.v2 as paddle
        
def data_reader(file_dir):
    def data_readers():
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + "/" + fi, "r") as f:
                for line in f:
                    query, title, label = line.strip('\n').split('\t')
                    if query == '' or title == '':
                        continue
                    query_data = []
                    title_data = []
                    num = len(query) + len(title)
                    found = 0
                    for word in query.split(" "):
                        try:
                            query_data.append(word_index[word])
                        except:
                            query_data.append(-1)
                    for word in title.split(" "):
                        try:
                            title_data.append(word_index[word])
                        except:
                            query_data.append(-1)
                    if len(query_data) == 0 or len(title_data) == 0:
                        continue
                    yield query_data, title_data, int(label)
    return data_readers
    
def fc_net(dict_size, class_dim=2, emb_dim=28, hid_dim=128):
    
    # input layers
    query_data = paddle.layer.data("query_word",
                             paddle.data_type.integer_value_sequence(dict_size))
    title_data = paddle.layer.data("title_word",
                             paddle.data_type.integer_value_sequence(dict_size))
    label = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))

    # embedding layer
    query_emb = paddle.layer.embedding(input=query_data, size=emb_dim)

    # pooling layers
    query_pool = paddle.layer.pooling(
        input=query_emb, pooling_type=paddle.pooling.Max())
    
    fc_layer_size = [28, 8]
    fc_layer_init_std = [1.0 / math.sqrt(s) for s in fc_layer_size]
    
    query_fc1 = paddle.layer.fc(
        input=query_pool,
        size=fc_layer_size[0],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[0]))
    query_fc2 = paddle.layer.fc(
        input=query_fc1,
        size=fc_layer_size[1],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[1]))

    title_emb = paddle.layer.embedding(input=title_data, size=emb_dim)
    
    title_pool = paddle.layer.pooling(
        input=title_emb, pooling_type=paddle.pooling.Max())

    title_fc1 = paddle.layer.fc(
        input=title_pool,
        size=fc_layer_size[0],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[0]))
    title_fc2 = paddle.layer.fc(
        input=title_fc1,
        size=fc_layer_size[1],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=fc_layer_init_std[1]))

    
    # output layer
    merge_layer = paddle.layer.concat(input=[query_fc2, title_fc2])
    output = paddle.layer.fc(
        input=merge_layer,
        size=class_dim,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(initial_std=1.0 / math.sqrt(class_dim)))

    cost = paddle.layer.classification_cost(input=output, label=label)

    return cost, output, label

    
def cnn_net(dict_dim, class_dim=2, emb_dim=28, hid_dim=128, conv_dim=300):
    
    # input layers
    query_data = paddle.layer.data("query_word",
                             paddle.data_type.integer_value_sequence(dict_dim))
    title_data = paddle.layer.data("title_word",
                             paddle.data_type.integer_value_sequence(dict_dim))
    label = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))

    # embedding layer
    query_emb = paddle.layer.embedding(input=query_data, size=emb_dim)

    # convolution layers
    query_conv_3 = paddle.networks.sequence_conv_pool(
        input=query_emb, context_len=3, hidden_size=conv_dim)
    query_conv_4 = paddle.networks.sequence_conv_pool(
        input=query_emb, context_len=4, hidden_size=conv_dim)
    
    hd_layer_size = 128
    hd_layer_init_std = 1.0 / math.sqrt(hd_layer_size)
    query_hd1 = paddle.layer.fc(
        input=[query_conv_3, query_conv_4],
        size=hd_layer_size,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=hd_layer_init_std))

    title_emb = paddle.layer.embedding(input=title_data, size=emb_dim)
    
    title_conv_3 = paddle.networks.sequence_conv_pool(
        input=title_emb, context_len=3, hidden_size=conv_dim)
    title_conv_4 = paddle.networks.sequence_conv_pool(
        input=title_emb, context_len=4, hidden_size=conv_dim)

    title_hd1 = paddle.layer.fc(
        input=[title_conv_3, title_conv_4],
        size=hd_layer_size,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=hd_layer_init_std))
    
    # output layer
    merge_layer = paddle.layer.concat(input=[query_hd1, title_hd1])
    output = paddle.layer.fc(
        input=merge_layer,
        size=class_dim,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(initial_std=1.0 / math.sqrt(class_dim)))

    cost = paddle.layer.classification_cost(input=output, label=label)

    return cost, output, label

def rnn_helper(dict_size,
            input_name,
            emb_dim=28,
            hidden_size=128, 
            stacked_rnn_num=5, 
            rnn_type="lstm"):
    input = paddle.layer.data(name=input_name, type=paddle.data_type.integer_value_sequence(dict_size))
    input_emb = paddle.layer.embedding(input=input, size=emb_dim)
    
    if rnn_type == "lstm":
        for i in range(stacked_rnn_num):
            rnn_cell = paddle.networks.simple_lstm(
                    input=rnn_cell if i else input_emb, size=hidden_size)
    elif rnn_type == "gru":
        for i in range(stacked_rnn_num):
            rnn_cell = paddle.networks.simple_gru(
                    input=rnn_cell if i else input_emb, size=hidden_size)
    else:
        raise Exception("invalid rnn_type!")
        
    rnn_pool = paddle.layer.pooling(
            input=rnn_cell, 
            pooling_type=paddle.pooling.Avg(),
            agg_level=paddle.layer.AggregateLevel.TO_NO_SEQUENCE)

    output = paddle.layer.fc(input=rnn_pool, size=hidden_size, act=paddle.activation.Tanh())

    return output
        
def rnn_net(dict_size,
        class_dim=2,
        emb_dim=28, 
        hidden_size=64, 
        stacked_rnn_num=10, 
        rnn_type="lstm"):
    
    query_out = rnn_helper(dict_size, "query_word", emb_dim, hidden_size, stacked_rnn_num, rnn_type)
    title_out = rnn_helper(dict_size, "title_word", emb_dim, hidden_size, stacked_rnn_num, rnn_type)
    label = paddle.layer.data(name="label", type=paddle.data_type.integer_value(class_dim))
    
    # output layer
    merge_layer = paddle.layer.concat(input=[query_out, title_out])
    output = paddle.layer.fc(
        input=merge_layer,
        size=class_dim,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(initial_std=1.0 / math.sqrt(class_dim)))

    cost = paddle.layer.classification_cost(input=output, label=label)

    return cost, output, label

    
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
    cost, output, label = cnn_net(dict_size, class_dim)

    # create parameters
    parameters = paddle.parameters.create(cost)
    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
            learning_rate = 0.01,
            regularization=paddle.optimizer.L2Regularization(rate=0.001),
            model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # add auc evaluator
    paddle.evaluator.auc(input=output, label=label)

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

    feeding = {"query_word":0, "title_word":1, "label":2}
    print "start training"
    trainer.train(
            reader=train_reader,
            event_handler=event_handler,
            feeding=feeding,
            num_passes=num_passes)
    
    print("Training finished")


    
if __name__ == "__main__":


    train_files = "./train_data_dir/20171017/"
    test_files = "./test_data_dir/test_set/"

    word_index = {}



    with open("thirdparty/search_query_title/sort_word_dict_10_9_idx_100w", "r") as f:
        for line in f:
            pair = line.strip("\n").split("\t")
            word = pair[0]
            idx = int(pair[1])
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
    
