import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import torch.nn.functional as F

def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    names, categories = zip(*batch)
    names_padded = torch.nn.utils.rnn.pad_sequence(names, batch_first=True)
    categories_stacked = torch.stack(categories)
    return names_padded, categories_stacked
    
class NameDataset(Dataset):
    def __init__(self, category_dict: Dict[str, List[str]], categories: List[str], letters: str):
        self.category_dict = category_dict
        self.categories = categories
        self.letters = letters

    def _one_hot_tensor(self, string: str) -> torch.Tensor:
        tensor = torch.zeros(len(string), 1, len(self.letters))
        for i, char in enumerate(string):
            tensor[i, 0, self.letters.index(char)] = 1
        return tensor
    
    def __len__(self) -> int:
        return sum(len(self.category_dict[category]) for category in self.categories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        category = self.categories[idx % len(self.categories)]
        category_tensor = torch.tensor(self.categories.index(category), dtype=torch.long)
        
        category_data = self.category_dict[category]

        names = category_data[idx % len(category_data)]
        names_tensor = self._one_hot_tensor(names)
    
        return names_tensor, category_tensor