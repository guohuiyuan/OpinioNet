#!/usr/bin/env python3
"""
将Train_reviews.csv和Train_labels.csv分割为训练集和验证集
保持同一个id的所有数据在同一个集合中
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_data():
    # 读取数据
    print("正在读取数据...")
    reviews_df = pd.read_csv('data/TRAIN/Train_reviews.csv')
    labels_df = pd.read_csv('data/TRAIN/Train_labels.csv')
    
    print(f"Reviews数据形状: {reviews_df.shape}")
    print(f"Labels数据形状: {labels_df.shape}")
    
    # 获取所有唯一的id
    unique_ids = reviews_df['id'].unique()
    print(f"唯一ID数量: {len(unique_ids)}")
    
    # 分割id为训练集和验证集 (80%训练, 20%验证)
    train_ids, val_ids = train_test_split(
        unique_ids, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"训练集ID数量: {len(train_ids)}")
    print(f"验证集ID数量: {len(val_ids)}")
    
    # 根据id分割reviews数据
    train_reviews = reviews_df[reviews_df['id'].isin(train_ids)]
    val_reviews = reviews_df[reviews_df['id'].isin(val_ids)]
    
    # 根据id分割labels数据
    train_labels = labels_df[labels_df['id'].isin(train_ids)]
    val_labels = labels_df[labels_df['id'].isin(val_ids)]
    
    print(f"训练集Reviews: {train_reviews.shape}")
    print(f"验证集Reviews: {val_reviews.shape}")
    print(f"训练集Labels: {train_labels.shape}")
    print(f"验证集Labels: {val_labels.shape}")
    
    # 创建输出目录
    output_dir = 'data/SPLIT'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分割后的数据
    train_reviews.to_csv(f'{output_dir}/train_reviews.csv', index=False)
    val_reviews.to_csv(f'{output_dir}/val_reviews.csv', index=False)
    train_labels.to_csv(f'{output_dir}/train_labels.csv', index=False)
    val_labels.to_csv(f'{output_dir}/val_labels.csv', index=False)
    
    print(f"\n数据已保存到 {output_dir}/ 目录:")
    print(f"- train_reviews.csv ({train_reviews.shape[0]} 行)")
    print(f"- val_reviews.csv ({val_reviews.shape[0]} 行)")
    print(f"- train_labels.csv ({train_labels.shape[0]} 行)")
    print(f"- val_labels.csv ({val_labels.shape[0]} 行)")
    
    # 创建信息文件
    with open(f'{output_dir}/split_info.txt', 'w') as f:
        f.write(f"数据分割信息\n")
        f.write(f"==============\n")
        f.write(f"原始数据:\n")
        f.write(f"- Train_reviews.csv: {reviews_df.shape[0]} 行\n")
        f.write(f"- Train_labels.csv: {labels_df.shape[0]} 行\n")
        f.write(f"- 唯一ID数量: {len(unique_ids)}\n\n")
        f.write(f"分割比例: 80% 训练集, 20% 验证集\n")
        f.write(f"随机种子: 42\n\n")
        f.write(f"训练集:\n")
        f.write(f"- ID数量: {len(train_ids)}\n")
        f.write(f"- Reviews: {train_reviews.shape[0]} 行\n")
        f.write(f"- Labels: {train_labels.shape[0]} 行\n\n")
        f.write(f"验证集:\n")
        f.write(f"- ID数量: {len(val_ids)}\n")
        f.write(f"- Reviews: {val_reviews.shape[0]} 行\n")
        f.write(f"- Labels: {val_labels.shape[0]} 行\n")
    
    print(f"\n分割信息已保存到 {output_dir}/split_info.txt")
    
    return train_reviews, val_reviews, train_labels, val_labels

if __name__ == "__main__":
    split_data()
