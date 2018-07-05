
# coding: utf-8

# 1. Построить диаграмму (гистограмму) количества хитов для каждой cell/блока/детектора.
# 2. Далее строить всячески пространственные диаграммы.
# 3. Аналогично пункту 1, только построить диаграмму количества ложных хитов (шума) для ячейки/блока/детектора.
# 4. Построить диаграмму количества хитов на частицу. Попытаться построить зависимости количества хитов от заряда/начального положения/начального импульса.

# In[2]:


import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[3]:


particles_df = pd.read_csv('datasets/unarch/train_100_events/event000001000-particles.csv')
hits_df = pd.read_csv('datasets/unarch/train_100_events/event000001000-hits.csv')
cells_df = pd.read_csv('datasets/unarch/train_100_events/event000001000-cells.csv')
truth_df = pd.read_csv('datasets/unarch/train_100_events/event000001000-truth.csv')


# In[4]:


print(particles_df.shape)
particles_df.head(4)


# In[5]:


particles_df.plot(kind='scatter', x='py', y='pz')


# In[6]:


print(hits_df.shape)
hits_df.head(4)


# In[7]:


hits_df.head(800).plot(kind='scatter', x='x', y='y')


# In[8]:


print(cells_df.shape)
cells_df.head(4)


# In[9]:


print(truth_df.shape)
truth_df.head(4)


# Кажется, есть возможность простым поиском соответствующего particle_id в truth_df, восстановить как пространственную траекторию частицы (нашёл строчку с соотвествующим particle_id - взял координаты, нашёл следующую строку с этим же particle_id - взял координаты). Может быть, можно попробовать предсказывать координты следующего хита имея "на руках" координаты текущего, нескольки предыдущих (вплот до точки рождения), заряд, импульс (значения предыдуших импульсов в моменты столкновений - хитов). Ну и можно воостановить не только траекторию "в координатах", но и траекторию в "ячейках" детекторов - hit_id у нас то же в truth_df имеется, а по нему легко воостановить ячейку/блок/детектор. А далее можно взять из таблицы детекторов координаты/матрицу поворота, и например попробовать найти завивисимости чего либо (например количества хитов) от характеристик блока/поворота. Или, ещё лучше - попробовать предсказыать импульс после столкновения, в зависимости от поворота, заряда, импульса в момент столкновения. 

# In[10]:


detectors_df = pd.read_csv('datasets/unarch/detectors.csv')


# In[11]:


print(detectors_df.shape)
detectors_df.head(4)


# In[12]:


submission_df = pd.read_csv('datasets/unarch/sample_submission.csv')


# In[13]:


print(submission_df.shape)
submission_df.head(4)


# In[14]:


train_dataset_dir = 'datasets/unarch/train_100_events'
dataset_filenames = os.listdir(train_dataset_dir)
print(dataset_filenames[:4])
event_ids = []
event_filenames = {}
for filename in dataset_filenames[:20]:
    event_id = filename[5:14]
    print(event_id)
    if event_id not in event_ids:
        event_ids.append(event_id)
    if event_id not in event_filenames:
        event_filenames[event_id] = [filename]
    else:
        event_filenames[event_id].append(filename)


# In[15]:


def random_sublist_select(original_list, sublist_size):
    return [original_list[i] for i in random.sample(range(len(original_list)), sublist_size)]


# In[16]:


def offset_sublist_select(original_list, sublist_size, offset):
    return original_list[offset:offset + sublist_size]


# In[17]:


def select_random_indexses_subset(size, subset_size):
    return random.sample(tuple(range(size)), subset_size) 


# In[18]:


def select_offset_indexses_subset(size, subset_size, offset):
    return tuple(range(size))[offset:offset + subset_size]


# In[19]:


def read_dataset_filenames_from_dir(path_to_datasets_dir):
    dataset_filenames = os.listdir(path_to_datasets_dir)
    event_filenames = {}
    for filename in dataset_filenames:
        path_to_file = os.path.join(path_to_datasets_dir, filename)
        event_id = filename[5:14]
        if event_id not in event_filenames:
            event_filenames[event_id] = [path_to_file]
        else:
            event_filenames[event_id].append(path_to_file)
    return event_filenames


# In[20]:


def select_events(indexes_list, event_names):
    return tuple(event_names[i] for i in indexes_list)


# In[21]:


def random_events_select(event_names, subset_size):
    event_names_len = len(event_names)
    indexes = select_random_indexses_subset(event_names_len, subset_size)
    return select_events(indexes, event_names)


# In[22]:


def offset_events_select(event_names, subset_size, offset):
    event_names_len = len(event_names)
    indexes = select_offset_indexses_subset(event_names_len, subset_size, offset)
    return select_events(indexes, event_names)


# In[23]:


def read_dataset_filenames_random(directory_list, sample_size=0):
    event_grouped_dataset_filenames = {}
    for directory in directory_list:
        dataset_filenames = read_dataset_filenames_from_dir(directory)
        event_names = tuple(dataset_filenames)
        if sample_size > 0:
            event_names = random_events_select(event_names, sample_size)
        dataset_filenames = {event_name: sorted(dataset_filenames[event_name]) for event_name in event_names}
        event_grouped_dataset_filenames.update(dataset_filenames)
    return event_grouped_dataset_filenames    


# In[24]:


def read_dataset_filenames_offset(directory_list, sample_size=0, offset=0):
    event_grouped_dataset_filenames = {}
    for directory in directory_list:
        dataset_filenames = read_dataset_filenames_from_dir(directory)
        event_names = tuple(dataset_filenames)
        if sample_size > 0:
            event_names = offset_events_select(tuple(dataset_filenames), sample_size, offset)
        dataset_filenames = {event_name: sorted(dataset_filenames[event_name]) for event_name in event_names}
        event_grouped_dataset_filenames.update(dataset_filenames)
    return event_grouped_dataset_filenames 


# In[25]:


def random_select_dataset_filenames(dataset_filenames, sample_size):
    return {
        event_name: dataset_filenames[event_name] for event_name in random_events_select(
            tuple(dataset_filenames),
            sample_size
        )
    }


# In[26]:


def offset_select_dataset_filenames(dataset_filenames, sample_size, offset):
    return {
        event_name: dataset_filenames[event_name] for event_name in offset_events_select(
            tuple(dataset_filenames),
            sample_size,
            offset
        )
    }


# In[27]:


def reduce_df_types(df):
    for column in df.columns:
        if df[column].dtype == np.float64:
            df.loc[:, column] = df[column].astype(np.float32)
        elif df[column].dtype == np.int64:
            df.loc[:, column] = df[column].astype(np.int32)
    return df


# In[28]:


event_grouped_dataset_filenames = read_dataset_filenames_random(['datasets/unarch/train_1/'], 10)


# In[29]:


print(event_grouped_dataset_filenames)


# In[30]:


#event_grouped_dataset_filenames_1 = read_dataset_filenames_offset(['datasets/unarch/train_1/'], 20)


# In[31]:


#event_grouped_dataset_filenames_1


# In[32]:


#all_event_grouped_dataset_filenames = read_dataset_filenames_random(['datasets/unarch/train_1/'])


# In[33]:


#random_selected_dataset_filenames = random_select_dataset_filenames(all_event_grouped_dataset_filenames, 100)


# In[34]:


#print(len(random_selected_dataset_filenames))
#random_selected_dataset_filenames


# In[35]:


def create_event_df(event_id, particles_df, truth_df, cells_df, hits_df):
    #return pd.merge(truth_df, hits_df, on='hit_id')
    truth_hits_df = pd.merge(truth_df, hits_df, on='hit_id')
    particles_truth_hits_df = pd.merge(particles_df, truth_hits_df, on='particle_id', how='right')
    #return particles_truth_hits_df
    return pd.merge(cells_df, particles_truth_hits_df, on='hit_id', how='outer')


# In[36]:


def read_dataset_to_grouped_by_event_dfs(selected_dataset_filenames):
    event_dfs = {}
    for event_id, event_filenames in selected_dataset_filenames.items():
        event_dfs[event_id] = create_event_df(
            event_id,
            pd.read_csv(event_filenames[2]),
            pd.read_csv(event_filenames[3]),
            pd.read_csv(event_filenames[0]),
            pd.read_csv(event_filenames[1])
        )
    return event_dfs        


# In[37]:


def read_dataset_to_grouped_by_event_dfs(selected_dataset_filenames):
    grouped_by_event_dfs = {}
    for event_id, event_filenames in selected_dataset_filenames.items():
        grouped_by_event_dfs[event_id] = (
            pd.read_csv(event_filenames[2]),
            pd.read_csv(event_filenames[3]),
            pd.read_csv(event_filenames[0]),
            pd.read_csv(event_filenames[1])
        )
    return grouped_by_event_dfs 


# In[38]:


def read_dataset_to_event_dfs(grouped_by_event_dfs):
    event_dfs = {}
    for event_id, dfs in grouped_by_event_dfs.items():
        event_dfs[event_id] = create_event_df(
            event_id,
            dfs[0],
            dfs[1],
            dfs[2],
            dfs[3]
        )
    return event_dfs 


# In[39]:


all(hits_df['hit_id'] == truth_df['hit_id'])


# In[40]:


event_df = create_event_df(0, particles_df, truth_df, cells_df, hits_df)


# In[41]:


event_df.shape


# In[42]:

random.seed(0)
grouped_by_event_datasets_df = read_dataset_to_grouped_by_event_dfs(event_grouped_dataset_filenames)


# In[43]:


grouped_by_event_datasets_df.keys()


# In[44]:


#grouped_by_event_datasets_df['000002231']


# In[45]:
'''
print("Before reduce types")
for event_id, dataframes in grouped_by_event_datasets_df.items():
    print(dataframes[0].info())
    print(dataframes[1].info())   
    print(dataframes[2].info())   
    print(dataframes[3].info())   
'''
'''
for event_id, dataframes in grouped_by_event_datasets_df.items():
    reduce_df_types(dataframes[0])
    reduce_df_types(dataframes[1])
    reduce_df_types(dataframes[2])
    reduce_df_types(dataframes[3])
    #print(dataframes[0].info())
    #print(dataframes[1].info())   
'''

'''
print("After reduce types")
for event_id, dataframes in grouped_by_event_datasets_df.items():
    print(dataframes[0].info())
    print(dataframes[1].info())   
    print(dataframes[2].info())   
    print(dataframes[3].info())   
'''

'''
for event_id, dataframes in grouped_by_event_datasets_df.items():
    print("\n\n", event_id)
    print(dataframes[0].info(), dataframes[1].info(), dataframes[2].info(), dataframes[3].info())   
'''

print("len(grouped_by_event_datasets_df): ", len(grouped_by_event_datasets_df))
# In[ ]:


## event_dfs = read_dataset_to_event_dfs(grouped_by_event_datasets_df)
grouped_by_event_datasets_df_part = {
        #event_id: dataframes for event_id, dataframes in list(grouped_by_event_datasets_df.items())[:8]
        event_id: dataframes for event_id, dataframes in grouped_by_event_datasets_df.items()
    }

'''
grouped_by_event_datasets_df_part = {
        #event_id: dataframes for event_id, dataframes in list(grouped_by_event_datasets_df.items())[:8]
        event_id: dataframes for event_id, dataframes in tuple(grouped_by_event_datasets_df.items())[:2]
    }
'''

event_dfs = read_dataset_to_event_dfs(grouped_by_event_datasets_df_part)

#print(list(event_dfs.values[0]).info())

for event_id, df in event_dfs.items():
    print(event_id)
    print(df.info())
    df.to_csv('{}.csv'.format(event_id, index=False))


# In[1]:


#event_dfs['000002771'].info()


# In[ ]:


## len(event_dfs)


# In[49]:


#not_merged_dfs_0 = grouped_by_event_datasets_df['000002771']


# In[50]:


#event_cells_0 = not_merged_dfs_1[0]


# In[56]:


#event_cells_0.isnull()


# In[54]:


def is_null(df):
    columns_dict = {}
    for column_name in df.columns.tolist():
        columns_dict[column_name] = any(df[column_name].isnull())
    return columns_dict


# In[61]:

'''
for event_id, grouped_dfs in grouped_by_event_datasets_df.items():
    print(event_id)
    print("particles:", is_null(grouped_dfs[0]))
    print("truth:", is_null(grouped_dfs[1]))
    print("cells:", is_null(grouped_dfs[2]))
    print("hits:", is_null(grouped_dfs[3]))
    
'''


# In[59]:

'''
for event_id, df in event_dfs.items():
    print(event_id)
    print(is_null(df))
'''

# In[62]:

'''
for event_id, grouped_dfs in grouped_by_event_datasets_df.items():
    print(event_id)
    print("particles:", grouped_dfs[0].info())
    print("truth:", grouped_dfs[1].info())
    print("cells:", grouped_dfs[2].info())
    print("hits:", grouped_dfs[3].info())
'''

# In[ ]:
