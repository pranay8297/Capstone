from sklearn.model_selection import train_test_split
from PIL import Image

from ipdb import set_trace as st

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class IAMDataset(Dataset):
    def __init__(self, df, root_dir, processor, max_target_length = 128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        rec = df.loc[idx]
        file_name = rec.file_name
        text = rec.text
        
        img = Image.open(f"{self.root_dir}/{file_name}").convert('RGB')
        pixel_values = self.processor(img, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        
        # important: make sure that PAD tokens are ignored by the loss function

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        return {'pixel_values': pixel_values.squeeze(), 'labels': torch.tensor(labels)}

def prepare_dataloaders(df, valid_size, image_dir, batch_size = 4):

    train_df, valid_df = train_test_split(df, test_size = valid_size)

    train_df.reset_index(drop = True, inplace = True)
    valid_df.reset_index(drop = True, inplace = True)

    train_ds = IAMDataset(train_df, image_dir, processor)
    valid_ds = IAMDataset(valid_df, image_dir, processor)

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size = batch_size)

    return train_loader, valid_loader

