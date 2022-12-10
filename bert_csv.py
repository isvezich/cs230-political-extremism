import os
import itertools
import csv
import gzip

import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

#IN_DIR = './x'
IN_DIR = './mydata_txt'
OUT_FILE = 'bert.csv.gz'

def partition_all(n, iterable):
    next = list(itertools.islice(iterable, n))
    while next:
        yield next
        next = list(itertools.islice(iterable, n))

def words(s):
    for w in s.split("."):
        yield w

def sentences():
    for dir, _subdirs, files in os.walk(IN_DIR):
        for file_name in files:
            # dir:       ./mydata_txt/1/bb8c56a9bf7fe46a0603a29180bf27d9b143c135
            # file_name: 1520969743_8470ud.txt
            with open(os.path.join(dir, file_name), 'r') as f:
                yield (f.read(), 0, dir, file_name)
                # for i, sentence in enumerate(sent_tokenize(f.read())):
                    # yield (sentence, i, dir, file_name)


def write_csv_gz():
    with gzip.open(OUT_FILE, 'wt') as f:
        w = csv.writer(f)
        w.writerow(['q_level', 'author', 'created_utc', 'post_id', 'sentence_i', 'text'])
        for (sentence, i, dir, file_name) in sentences():
            parts = dir.split('/')
            author = parts[3]
            q_level = parts[2]
            fparts = file_name.split('_')
            ts = fparts[0]
            id = fparts[1].removesuffix(".txt")
            w.writerow([q_level, author, ts, id, i, sentence])


write_csv_gz()

