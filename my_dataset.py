import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, max_len, pad_index=0, bos_index=2, eos_index=3):
            
        super().__init__()
        
        self.data = data
        self.length = len(self.data)
        
        self.max_len = max_len
        
        self.pad_index = pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index

    def __len__(self):
        
        return self.length

    def __getitem__(self, index):
        
        sequence = [self.bos_index] + self.data[index]
        sequence = sequence[:self.max_len]
        
        x = sequence[:]
        y = sequence[1:] + [self.eos_index]
        true_length = torch.tensor(len(x))
        
        assert len(x) == len(y)
        
        pads = [self.pad_index] * (self.max_len - len(x))
        
        x = torch.tensor(x + pads).long()
        y = torch.tensor(y + pads).long()
        
        
        return x, y, true_length