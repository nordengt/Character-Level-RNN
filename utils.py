import torch
from torch import Tensor
from typing import Tuple, List, Optional

import unicodedata
import matplotlib.pyplot as plt

def unicode_to_ascii(name: str, letters: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", name)
        if not unicodedata.combining(c) and c in letters
    )

def name_to_tensor(name: str, letters: str, n_letters: int) -> Tensor:
    name = unicode_to_ascii(name, letters)
    name_tensor = torch.zeros(len(name), 1, n_letters)
    for i, char in enumerate(name):
        name_tensor[i, 0, letters.index(char)] = 1
    return name_tensor

def category_to_tensor(category: str, categories: list, n_categories: int) -> Tensor:
    idx = categories.index(category)
    cat_tensor = torch.zeros(1, n_categories)
    cat_tensor[0][idx] = 1
    return cat_tensor

def target_to_tensor(name: str, letters: str, n_letters: int) -> Tensor:
    letter_indexes = [letters.find(name[li]) for li in range(1, len(name))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def output_category(output: torch.Tensor, categories: list) -> Tuple[str, int]:
    return categories[torch.argmax(output, dim=1).item()]

def plot_losses(losses: List[float], save_path: Optional[str] = None) -> None:
    plt.plot(losses, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()