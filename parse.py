# -*- coding: gbk -*-

import sys
import random
import wordseg
import urllib
import random
    
pos = []
for line in sys.stdin:
    line = line.strip("\n").split("\t")
    query = line[2]
    title = line[5]
    if query == "" or title == "":
        continue
    query = urllib.unquote(query)
    title = urllib.unquote(title)
    label = int(line[7])
    pos.append([query, title, label])


strip_chars = [".", ",", "-", "_", ":"]
final = []

MAX_TERM_COUNT = 1024
    
dict_handle = wordseg.scw_load_worddict("./dict/wordseg_dict/")
result_handle = wordseg.scw_create_out(MAX_TERM_COUNT)
token_handle = wordseg.create_tokens(MAX_TERM_COUNT)
token_handle = wordseg.init_tokens(token_handle, MAX_TERM_COUNT) 

for query, title, label in pos:
    
    for char in strip_chars:
        query = query.strip(char)
        title = title.strip(char)
    
    query_title = []
    for line in [query, title]:
        wordseg.scw_segment_words(dict_handle, result_handle, line, 1)
        token_count = wordseg.scw_get_token_1(result_handle, wordseg.SCW_WPCOMP, token_handle, MAX_TERM_COUNT) 
        query_title.append([token[7] for token in wordseg.tokens_to_list(token_handle, token_count)])

    query = " ".join(query_title[0])
    title = " ".join(query_title[1])
    final.append([query, title, label]) 

wordseg.destroy_tokens(token_handle)
wordseg.scw_destroy_out(result_handle)
wordseg.scw_destroy_worddict(dict_handle)

for query, title, label in final:
    _, ti, _ = random.choice(pos)
    if random.randint(0, 1) == 1:
        print "\t".join(map(str, [query, ti, title, 0]))
    else:
        print "\t".join(map(str, [query, title, ti, 1]))


