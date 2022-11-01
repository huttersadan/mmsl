import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import os
from utils.basic_utils import BigFile

def average_to_fixed_length1(visual_input,map_size):
    #输入是一个length(frames) * 1024维的向量
    #用来对frame feature进行mean pooling
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size  #这是取样间隔
    num_clips = visual_input.shape[0]# 这是frame的个数
    idxs = torch.arange(0,num_sample_clips+1,1.0)/num_sample_clips*num_clips# 将frame分成32份
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx,e_idx = idxs[i].item(),idxs[i+1].item() #第i份feature，里面包含num_sample_clips个feature，然后对这些feature进行平均操作
        if s_idx<e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))#这个意思是从s_idx到e_idx的所有frame进行一个平均
        else:#相等的情况
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input,dim=0).numpy()
    
    return new_visual_input



def l2_normalize_np_array1(np_array,eps =1e-10):
    #注意到，这里要让最后一个维度正则化一下，第一个维度是map_size
    return np_array/(np.linalg.norm(np_array,axis=-1,keepdims = True)+eps)
    
def uniform_feature_sampling1(features,max_len):
    #输入的feature是frame-list
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    # 我感觉和clip是一样的，无外乎frame的尺度更大，取样间隔更短。
    idxs= np.arange(0,max_len+1,1.0)/max_len*num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs>num_clips-1] = num_clips -1
    new_features = []
    for i in range(max_len):
        s_idx,e_idx = idxs[i],idxs[i+1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx],axis = 0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features


    




class Dataset4MS_SL(data.Dataset):
    def __init__(self,opt):
        self.root_path = opt.root_path
        self.visual_feature = opt.visual_feature
        self.collection = opt.collection
        self.visual_root_path =  os.path.join(self.root_path,self.collection,"FeatureData",self.visual_feature)
        self.text_root_path = os.path.join(self.root_path,self.collection,'TextData')
        self.text_caption_txt = os.path.join(self.text_root_path,self.collection+"train.caption.txt")
        self.text_caption_hdf5 = os.path.join(self.text_root_path,"roberta_"+self.collection+"_query_feat.hdf5")
        text_cap_ids_list = [] #用来储存text_cap_id的list
        text_ids_caption_dict ={}#用来对应储存text_cap_id和text_caption的dict
        video_id_list = []#用来储存video_id的list
        video_id_capid_dict = {}#用来储存video对应的text_caption的字典，字典的key是video的id，字典的值是一个list，list中存储着多个对应的text caption
        with open(self.text_caption_txt,'r') as f:
            for line in f.readlines():
                text_id,text_caption = line.strip().split(' ',1)
                text_ids_caption_dict[text_id] = text_caption
                text_cap_ids_list.append(text_id)
                video_id = text_id.split('#')[0]
                if video_id not in video_id_list:
                    #如果这个video_id第一次出现
                    video_id_list.append(video_id)
                if video_id not in video_id_capid_dict:
                    #如果这个video_id第一次出现
                    video_id_capid_dict[video_id] = []
                    video_id_capid_dict[video_id].append(text_id)
                else:
                    #不是第一次出现
                    video_id_capid_dict[video_id].append(text_id)
        self.text_cap_ids_list = text_cap_ids_list
        self.text_ids_caption_dict = text_ids_caption_dict
        self.video_id_list = video_id_list
        self.video_id_capid_dict = video_id_capid_dict
        self.video_length = len(self.video_id_list)
        
        # frames id
        self.video2frames_path = os.path.join(self.visual_root_path,'video2frames.txt')
        with open(self.video2frames_path,'r') as f:
            video_frames_dict = eval(f.read())
        self.video_frames_dict = video_frames_dict

        # frames feature
        self.frames_feature_path = os.path.join(self.visual_root_path,'feature.bin')
        self.frames_feature_BigFile = BigFile(self.visual_root_path)

        #opt
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l
        self.open_file = False
        
        self.length = len(self.video_id_list)
        
    def __getitem__(self,index):
        if self.open_file:
            self.open_file = True
        else:
            self.text_feature_file = h5py.File(self.text_caption_hdf5,'r')
            self.open_file = True
        #要提取video和对应的text cap
        video_id = self.video_id_list[index]
        text_ids = self.video_id_capid_dict[video_id]#这是包含了所有text_id的list

        #先处理video
        frame_ids = self.video_frames_dict[video_id]
        frame_feature_list = []#由于frame是多个的，所以要用一个list存储
        
        for frame_id in frame_ids:
            frame_feature_list.append(self.frames_feature_BigFile.read_one(frame_id))

        #生成clip feature和 frame feature
        clip_video_feature = average_to_fixed_length1(np.array(frame_feature_list),self.map_size)
        clip_video_feature = l2_normalize_np_array1(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
        
        frame_video_feature = uniform_feature_sampling1(np.array(frame_feature_list),self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array1(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)


        #再处理text
        text_tensor_list = []
        for text_id in text_ids:
            text_feature = self.text_feature_file[text_id][...]
            text_feature_tensor = torch.from_numpy(l2_normalize_np_array1(text_feature))[:self.max_desc_len]
            text_tensor_list.append(text_feature_tensor)
        return clip_video_feature,frame_video_feature,text_tensor_list,index,text_ids,video_id
    def __len__(self):
        return self.length


class VisDataSet4MS_SL(data.Dataset):
    def __init__(self,opt,video_ids = None):
        self.root_path = opt.root_path
        self.visual_feature = opt.visual_feature
        self.collection = opt.collection
        self.visual_root_path =  os.path.join(self.root_path,self.collection,"FeatureData",self.visual_feature)
        self.video2frames_path = os.path.join(self.visual_root_path,'video2frames.txt')
        with open(self.video2frames_path,'r') as f:
            video_frames_dict = eval(f.read())
        self.video_frames_dict = video_frames_dict

        self.frames_feature_path = os.path.join(self.visual_root_path,'feature.bin')
        self.frames_feature_BigFile = BigFile(self.visual_root_path)

        
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = self.video_frames_dict.keys()
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
    def __getitem__(self,index):
        video_id = self.video_ids[index]
        frame_list = self.video_frames_dict[video_id]
        frame_feature_list = []
        for frame_id in frame_list:
            frame_feature_list.append(self.frames_feature_BigFile.read_one(frame_id))
        
        clip_video_feature = average_to_fixed_length1(np.array(frame_feature_list),self.map_size)
        clip_video_feature = l2_normalize_np_array1(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
        
        frame_video_feature = uniform_feature_sampling1(np.array(frame_feature_list),self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array1(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        return clip_video_feature,frame_video_feature,index,video_id

    def __len__(self):
        return self.length
    

class TxtDataSet4MS_SL(data.Dataset):
    def __init__(self,opt,type_of_dataset = 'val'):
        self.root_path = opt.root_path
        self.visual_feature = opt.visual_feature
        self.collection = opt.collection
        self.type_of_dataset = type_of_dataset
        self.visual_root_path =  os.path.join(self.root_path,"FeatureData",self.visual_feature)
        self.text_root_path = os.path.join(self.root_path,self.collection,'TextData')
        self.text_caption_txt = os.path.join(self.text_root_path,self.collection+self.type_of_dataset+".caption.txt")
        self.text_caption_hdf5 = os.path.join(self.text_root_path,"roberta_"+self.collection+"_query_feat.hdf5")
        text_cap_ids_list = [] #用来储存text_cap_id的list
        text_ids_caption_dict ={}#用来对应储存text_cap_id和text_caption的dict
        #video_id_list = []#用来储存video_id的list
        #video_id_capid_dict = {}#用来储存video对应的text_caption的字典，字典的key是video的id，字典的值是一个list，list中存储着多个对应的text caption
        with open(self.text_caption_txt,'r') as f:
            for line in f.readlines():
                text_id,text_caption = line.strip().split(' ',1)
                text_ids_caption_dict[text_id] = text_caption
                text_cap_ids_list.append(text_id)
        self.text_ids_caption_dict=  text_ids_caption_dict
        self.text_cap_ids_list = text_cap_ids_list
        self.max_desc_len = opt.max_desc_l
        self.open_file = False
        self.length = len(self.text_cap_ids_list)
        
    def __getitem__(self,index):
        if self.open_file:
            self.open_file = True
        else:
            self.text_feature_file = h5py.File(self.text_caption_hdf5,'r')
            self.open_file = True
        text_id = self.text_cap_ids_list[index]
        text_feature = self.text_feature_file[text_id][...]
        text_tensor = torch.from_numpy(l2_normalize_np_array1(text_feature))[:self.max_desc_len]
        return text_tensor,index,text_id
    def __len__(self):
        return self.length

# root_path = "/data/dtt/mssl/dataset/charades"
# visual_feature = "i3d_rgb_lgi"
# collection = "charades"
# visual_root_path = os.path.join(root_path,"FeatureData",visual_feature)
# text_root_path = os.path.join(root_path,'TextData')
# #print(visual_root_path,text_root_path)
# # text caption:txt    roberta_charades. hdf5        
# text_caption_txt = os.path.join(text_root_path,collection+"train.caption.txt")
# text_caption_hdf5 = os.path.join(text_root_path,"roberta_charades_query_feat.hdf5")
# text_cap_ids_list = [] #用来储存text_cap_id的list
# text_ids_caption_dict ={}#用来对应储存text_cap_id和text_caption的dict
# video_id_list = []#用来储存video_id的list
# video_id_capid_dict = {}#用来储存video对应的text_caption的字典，字典的key是video的id，字典的值是一个list，list中存储着多个对应的text caption
# with open(text_caption_txt,'r') as f:
#     for line in f.readlines():
#         text_id,text_caption = line.strip().split(' ',1)
#         text_ids_caption_dict[text_id] = text_caption
#         vedio_id = text_id.split('#')[0]
#         if video_id not in video_id_list:
#             #如果这个video_id第一次出现
#             video_id_list.append(video_id)
#         if video_id not in video_id_capid_dict:
#             #如果这个video_id第一次出现
#             video_id_capid_dict[video_id] = []
#             video_id_capid_dict[video_id].append(cap_id)
#         else:
#             #不是第一次出现
#             video_id_capid_dict[video_id].append(cap_id)
#text_feat = h5py.File(text_caption_hdf5,'r')

# print(text_feat)
# #key 是 cap_id
# for k in text_feat.keys():
#     k_dataset = text_feat[k]
#     print(k_dataset[...] == k_dataset)

# video2frames_path = '/data/dtt/mssl/dataset/charades/FeatureData/i3d_rgb_lgi/video2frames.txt'
# with open(video2frames_path,'r') as f:
#     video_frames_dict = eval(f.read())
#     print(video_frames_dict.keys())

#frames_feature_path = os.path.join(visual_root_path,'feature.bin')
#frame_feature_BigFile = BigFile(visual_root_path)
#print(np.array(frame_feature_BigFile.read_one('00607_70')).shape)

# a = torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15])
# b = torch.tensor(15)
# c = torch.min(a,b)
# print(c)
# print(c.shape)

# x = torch.randn(2, 3)
# x = [x,x]
# print(torch.cat(x,dim= 0))
def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids = zip(*data)

    #videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0



    return dict(clip_video_features=clip_videos,
                frame_video_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=target,
                text_mask=words_mask,
                text_labels=labels
                )
    
def collate_frame_val(data):
    #frame_video_features : num_of_videos X  frames in one video   X  frame features
    clip_video_features,frame_video_features,idxs,video_ids = zip(*data)
    clip_videos = torch.cat(clip_video_features,dim =0 ).float()
    video_lengths = [len(frame) for frame in frame_video_features]# 每个视频的长度
    frame_vec_len = len(frame_video_features[0][0])# frame 的feature的个数
    frame_videos = torch.zeros(len(frame_video_features),max(video_lengths),frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features),max(video_lengths))
    for i,frames in enumerate(frame_video_features):
        end = video_lengths[i] #第i个视频的长度，有几个frame
        frame_videos[i,:end,:] = frames[:end,:]#这里的end可以直接去掉
        videos_mask[i,:end] = 1
    return clip_videos,frame_videos,videos_mask,idxs,video_ids

def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key = lambda x:len(x[0]),reverse = True)
    captions,idxs ,cap_ids = zip(*data)
    if captions[0] is not None:
        lengths = [len(cap) for cap in captions]#每个caption的长度
        target = torch.zeros(len(captions),max(lengths),captions[0].shape[-1])# n个caption ，最长的长度m，以及单词的embedding 维度h
        words_mask = torch.zeros(len(captions),max(lengths))# 掩盖哪些单词
        for i,cap in enumerate(captions):
            end=  lengths[i]
            target[i,:end] = cap[:end]
            words_mask[i,:end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None
    return target,words_mask,idxs,cap_ids


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list