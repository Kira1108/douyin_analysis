import pymysql
import pandas as pd
import numpy as np
import json

def parse_region_rate(x):
    dist = json.loads(x)
    if len(dist) == 0:
        return None
    
    result = {}
    for city in dist:
        for k,v in city.items():
            result[k] = float(v[:-1]) / 100
    return result

def get_all_cities(data):
    all_cities = []
    for record in data.region_rate.values.tolist():
        if record is None:
            continue
        for k,v in record.items():
            all_cities.append(k)
    return set(all_cities)

def make_region_vector(d, n_cities, city2id):
    zeros = np.zeros(n_cities)
    if d is None:
        return zeros
    dist = {city2id.get(k):v for k, v in d.items()}
    zeros[np.array(list(dist.keys()))] = np.array(list(dist.values()))
    region_vector = zeros.tolist()
    return region_vector

def parse_age_rate(a):
    a = json.loads(a)
    if not len(a) > 0:
        return None
    
    result = {}
    for age in a:
        result[age['age']] = age['count']
    return result

def get_all_ages(data):
    all_ages = []
    for age_dist in data.age_rate.values.tolist():
        if age_dist is None:
            continue

        for k,v in age_dist.items():
            all_ages.append(k)    
    return set(all_ages)

def make_age_vector(a, n_ages, age2id):
    zeros = np.zeros(n_ages)
    if a is None:
        return zeros
    
    for age, count in a.items():
        zeros[age2id[age]] = count
    return zeros.tolist()

def get_fans_counts(f, col = 'Count'):
    t = json.loads(f)
    
    if not len(t) > 0:
        return None
    counts= []
    for t in t:
        counts.append(t[col])
    return counts

def parse_word_cloud(w):
    w = json.loads(w)
    if not len(w) > 0:
        return None
    
    result = {}
    for i in w:
        result[i['name']] = i['value']
    return result

def get_vab(data):
    vocab = []
    for up in data.comment_cloud.values.tolist():
        if up is None:
            continue
        for k,v in up.items():
            vocab.append(k)
    return set(vocab)

def make_word_vector(c, word2index, vocab_size):
    vector = np.zeros(vocab_size)
    if c is None:
        return vector

    for word, weight in c.items():
        vector[word2index.get(word)] = weight
    return vector

def make_weekly_vector(w,col = 'Count'):
    
    w = json.loads(w)
    vector = np.zeros(7)
    if len(w) !=7:
        return vector
    for day in w:
        if day['Origin'] != 0:
            ind = day['Origin'] - 1
        else:
            ind = 6

        vector[ind] = day[col]
    return vector


def fill_none_wt_zero_lists(df, col, size):
    res = []
    for l in df[col].values:
        if l is None:
            res.append([0]*size)
        else:
            res.append(l)
    df[col] = res    
    return df