from train_cluster import *

cluster_test_dir = "./test_data_dir/test_set/"
file_name = "./result/param_cnn_final.tar.gz"

print "loading dict"

word_index = {}

with open("thirdparty/search_query_title/sort_word_dict_10_9_idx_100w", "r") as f:
    for line in f:
        word, index = line.strip("\n").split("\t")
        word_index[word] = index

dict_dim = len(word_index)
class_dim = 2


print("building cnn")
paddle.init(use_gpu=False)
_, output, _ = cnn_net(dict_dim, class_dim=class_dim)
"""
print("loading params")

with gzip.open(file_name) as f:
    parameters = paddle.parameters.Parameters.from_tar(f)
"""
queries = []
true_labels = []

print "reading data"

i = 0

files = os.listdir(cluster_test_dir)
for file in files:
    with open(cluster_test_dir + file) as f:    
        for line in f:
            stuff = line.strip("\n").split("\t")
            query = stuff[0]
            title = stuff[1]
            label = stuff[2]
            
            query_data = []
            title_data = []
            
            for word in query.split(" "):
                try:
                    query_data.append(word_index[word])
                except:
                    query_data.append(-1)
            for word in title.split(" "):
                try:
                    title_data.append(word_index[word])
                except:
                    title_data.append(-1)
                    
            queries.append((query_data, title_data))
            true_labels.append(label)
            if i < 10:
                print query
                print title
                print [query_data, title_data]
                print 
                i += 1
print "start infer..."
"""
predictions = paddle.infer(
    output_layer=output,
    parameters=parameters,
    input=queries)

with open("prediction.txt", "w") as f:
    for item in predictions:
        f.write(item)
        f.write("\n")
    
"""
