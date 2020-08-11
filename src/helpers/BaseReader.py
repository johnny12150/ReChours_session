# -*- coding: UTF-8 -*-

import os
import time
import pickle
import argparse
import logging
import numpy as np
import pandas as pd


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--history_max', type=int, default=20,
                            help='Maximum length of history.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self.history_max = args.history_max

        t0 = time.time()
        self._read_data()
        self._append_info()
        logging.info('Done! [{:<.2f} s]'.format(time.time() - t0) + os.linesep)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df, self.item_meta_df = dict(), pd.DataFrame()
        self._read_preprocessed_df()

        logging.info('Formating data type...')
        for df in list(self.data_df.values()) + [self.item_meta_df]:
            for col in df.columns:
                df[col] = df[col].apply(lambda x: eval(str(x)))

        logging.info('Constructing relation triplets...')
        self.triplet_set = set()
        relation_types = [r for r in self.item_meta_df.columns if r.startswith('r_')]
        heads, relations, tails = [], [], []
        for idx in range(len(self.item_meta_df)):
            head_item = self.item_meta_df['item_id'][idx]
            for r_idx, r in enumerate(relation_types):
                for tail_item in self.item_meta_df[r][idx]:
                    heads.append(head_item)
                    relations.append(r_idx + 1)
                    tails.append(tail_item)
                    self.triplet_set.add((head_item, r_idx + 1, tail_item))
        self.relation_df = pd.DataFrame()
        self.relation_df['head'] = heads
        self.relation_df['relation'] = relations
        self.relation_df['tail'] = tails

        logging.info('Counting dataset statistics...')
        self.all_df = pd.concat([self.data_df[key][['user_id', 'item_id', 'time']] for key in ['train', 'dev', 'test']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        self.n_relations = self.relation_df['relation'].max() + 1
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(self.n_users, self.n_items, len(self.all_df)))
        logging.info('"# relation": {}, "# triplet": {}'.format(self.n_relations, len(self.relation_df)))

    def _append_info(self):
        """
        Add history info to data_df: item_his, time_his, his_length
        ! Need data_df to be sorted by time in ascending order
        :return:
        """
        logging.info('Adding history info...')
        user_his_dict = dict()  # store the already seen sequence of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            i_history, t_history = [], []
            for uid, iid, t in zip(df['user_id'], df['item_id'], df['time']):
                if uid not in user_his_dict:
                    user_his_dict[uid] = []
                i_history.append([x[0] for x in user_his_dict[uid]])
                t_history.append([x[1] for x in user_his_dict[uid]])
                user_his_dict[uid].append((iid, t))
            df['item_his'] = i_history
            df['time_his'] = t_history
            if self.history_max > 0:
                df['item_his'] = df['item_his'].apply(lambda x: x[-self.history_max:])
                df['time_his'] = df['time_his'].apply(lambda x: x[-self.history_max:])
            df['his_length'] = df['item_his'].apply(lambda x: len(x))

        self.user_clicked_set = dict()
        for uid in user_his_dict:
            self.user_clicked_set[uid] = set([x[0] for x in user_his_dict[uid]])

    def _read_preprocessed_df(self):
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)

        item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
        if os.path.exists(item_meta_path):
            self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = BaseReader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    args.path = '../../data/'
    corpus = BaseReader(args)

    corpus_path = os.path.join(args.path, args.dataset, 'Corpus.pkl')
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(corpus, open(corpus_path, 'wb'))
