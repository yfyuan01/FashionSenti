# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals
cap_root = '../../data/b-t4sa/raw_tweets_text.csv'
with open(cap_root,'r') as f:
    cap_list = f.readlines()
cap_list = [c.strip('\n').split(',') for c in cap_list]
