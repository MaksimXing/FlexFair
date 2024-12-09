import os
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

def GetBCNDF(base_skin_dir, csv_name):
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(base_skin_dir, '*.[jJ][pP][gG]'))}

    tile_df = pd.read_csv(os.path.join(base_skin_dir, csv_name))
    tile_df['path'] = tile_df['isic_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['benign_malignant']
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

    tile_df['A_sex'] = tile_df['sex'].map({'male': 0, 'female': 1})
    tile_df['A_age'] = tile_df['age_approx'].apply(lambda x: 0 if x <= 60 else 1)
    # HAM0, BCN1
    tile_df['A_site'] = tile_df['attribution'].map({'ViDIR Group, Department of Dermatology, Medical University of Vienna': 0, 'Hospital Clínic de Barcelona': 1})

    tile_df[['cell_type_idx', 'cell_type', 'A_sex', 'A_age', 'A_site']].sort_values('cell_type_idx').drop_duplicates()

    return tile_df


class BCN_SkinDataset(data.Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))
        sex = int(self.df['A_sex'][index])
        age = int(self.df['A_age'][index])
        site = int(self.df['A_site'][index])
        path = self.df['path'][index]
        
        if self.transform:
            X = self.transform(X)
            
        return X, y, sex, age, site, path