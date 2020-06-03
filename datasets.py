#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:10:48 2020

@author: krishna
"""

import os
import glob
import argparse
import json



class TIMIT(object):
    def __init__(self,config):
        super(TIMIT, self).__init__()
        self.timit_root = config.timit_dataset_root
        self.store_path = config.timit_save_root
    
   
        
    def create_spk_mapping_train(self):
        spk_dict={}
        self.train_dir = os.path.join(self.timit_root,'TRAIN')
        train_subfolders = sorted(glob.glob(self.train_dir+'/*/'))
        spk_list=[]
        for sub_folder in train_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for speaker_folder in speaker_folders:  
                spk_name = speaker_folder.split('/')[-2]
                spk_list.append(spk_name)
        for i in range(len(spk_list)):
            spk_dict[spk_list[i]]=i
        return spk_dict
        
    def create_spk_mapping_test(self):
        spk_dict={}
        self.test_dir = os.path.join(self.timit_root,'TEST')
        test_subfolders = sorted(glob.glob(self.test_dir+'/*/'))
        spk_list=[]
        for sub_folder in test_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for speaker_folder in speaker_folders:  
                spk_name = speaker_folder.split('/')[-2]
                spk_list.append(spk_name)
        for i in range(len(spk_list)):
            spk_dict[spk_list[i]]=i
        return spk_dict
    
    
    
            
    def process_data_train(self):
        spk_dict = self.create_spk_mapping_train()
        self.train_dir = os.path.join(self.timit_root,'TRAIN')
        train_subfolders = sorted(glob.glob(self.train_dir+'/*/'))
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
        fid = open(os.path.join(self.store_path,'training.txt'),'w')
                
        for sub_folder in train_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for spk_folder in speaker_folders:
                WAV_files = sorted(glob.glob(spk_folder+'/*.WAV'))
                for audio_filepath in WAV_files:
                    spk_name = audio_filepath.split('/')[-2]
                    spk_id = spk_dict[spk_name]
                    to_write= audio_filepath+' '+str(spk_id)
                    fid.write(to_write+'\n')
            
    
        
    def process_data_test(self):
        spk_dict = self.create_spk_mapping_test()
        self.test_dir = os.path.join(self.timit_root,'TEST')
        test_subfolders = sorted(glob.glob(self.test_dir+'/*/'))
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
        fid = open(os.path.join(self.store_path,'evaluation.txt'),'w')
        
        for sub_folder in test_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for spk_folder in speaker_folders:
                WAV_files = sorted(glob.glob(spk_folder+'/*.WAV'))
                for audio_filepath in WAV_files:
                    print(audio_filepath)
                    spk_name = audio_filepath.split('/')[-2]
                    spk_id = spk_dict[spk_name]
                    to_write= audio_filepath+' '+str(spk_id)
                    fid.write(to_write+'\n')
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--timit_dataset_root", default="/media/newhd/TIMIT/data/lisa/data/timit/raw/TIMIT", type=str,help='Dataset path')
    parser.add_argument("--timit_save_root", default="meta", type=str,help='Save directory after processing')
    
    config = parser.parse_args()
    timit = TIMIT(config)
    timit.process_data_train()
    timit.process_data_test()