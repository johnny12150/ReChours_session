import pandas as pd
import numpy as np
import datetime as dt
import os

RAW_PATH = 'diginetica/'
file_path = 'F:/data/dataset-train-diginetica/train-clicks.csv'  # buy
file_path2 = 'F:/data/dataset-train-diginetica/train-item-views.csv'  # view
file_path3 = 'F:/data/dataset-train-diginetica/products.csv'
file_path4 = 'F:/data/dataset-train-diginetica/product-categories.csv'
df = pd.read_csv(file_path2, delimiter=';')
df['time'] = df['eventdate'].astype(str).apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp())
df['time'] += df['timeframe']
print(df.eventdate.min(), df.eventdate.max())  # duration is 5 months

item_meta = pd.read_csv(file_path3, delimiter=';')
item_meta2 = pd.read_csv(file_path4, delimiter=';')
item_meta = item_meta.merge(item_meta2, how='outer', on='itemId')
item_meta['r_complement'] = [[]]*item_meta.shape[0]
item_meta['r_substitute'] = [[]]*item_meta.shape[0]

# 找有商品資訊的item id (有出現在session內過的item)
useful_meta = item_meta[item_meta['itemId'].isin(df['itemId'])].reset_index(drop=True)
all_items = set(useful_meta.itemId.values.tolist())
df = df[df.itemId.isin(all_items)]

np.random.seed(2019)
NEG_ITEMS = 99
# 先用session id替代user
out_df = df.rename(columns={'itemId': 'item_id', 'sessionId': 'user_id'})
out_df = out_df[['user_id', 'item_id', 'time']]
out_df = out_df.drop_duplicates(['user_id', 'item_id', 'time'])
out_df.sort_values(by=['time', 'user_id', 'item_id'], inplace=True)

# reindex (start from 1)
uids = sorted(out_df['user_id'].unique())
user2id = dict(zip(uids, range(1, len(uids) + 1)))
iids = sorted(out_df['item_id'].unique())
item2id = dict(zip(iids, range(1, len(iids) + 1)))
out_df['user_id'] = out_df['user_id'].apply(lambda x: user2id[x])
out_df['item_id'] = out_df['item_id'].apply(lambda x: item2id[x])
out_df = out_df.reset_index(drop=True)

clicked_item_set = dict()
for user_id, seq_df in out_df.groupby('user_id'):
    clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())


def generate_dev_test(data_df):
    result_dfs = []
    for idx in range(2):
        result_df = data_df.groupby('user_id').tail(1).copy()
        data_df = data_df.drop(result_df.index)
        neg_items = np.random.randint(1, len(iids) + 1, (len(result_df), NEG_ITEMS))
        for i, uid in enumerate(result_df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                while neg_items[i][j] in user_clicked:
                    neg_items[i][j] = np.random.randint(1, len(iids) + 1)
        result_df['neg_items'] = neg_items.tolist()
        result_dfs.append(result_df)
    return result_dfs, data_df

leave_df = out_df.groupby('user_id').head(1)
data_df = out_df.drop(leave_df.index)

[test_df, dev_df], data_df = generate_dev_test(data_df)
train_df = pd.concat([leave_df, data_df]).sort_index()

train_df.to_csv(os.path.join(RAW_PATH, 'train.csv'), sep='\t', index=False)
dev_df.to_csv(os.path.join(RAW_PATH, 'dev.csv'), sep='\t', index=False)
test_df.to_csv(os.path.join(RAW_PATH, 'test.csv'), sep='\t', index=False)
item_meta = item_meta.rename(columns={'itemId': 'item_id'})
item_meta.to_csv(os.path.join(RAW_PATH, 'item_meta.csv'), sep='\t', index=False)
