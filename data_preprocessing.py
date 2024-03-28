import glob
from typing import List, Dict

from utils import unicode_to_ascii

def get_names(file: str, letters: str) -> List[str]:
    names = open(file, encoding="utf-8").read().strip().split("\n")
    return [unicode_to_ascii(name, letters) for name in names]

def get_data(path: str, letters: str) -> Dict[str, List[str]]:
    files = glob.glob(path)
    category_name_dict = {}
    for filename in files:
        category = filename.split("\\")[-1].split('.')[0]
        category_name_dict[category] = get_names(filename, letters)
    categories = list(category_name_dict.keys())
    n_letters = len(letters)
    n_categories = len(categories)
    return category_name_dict, categories, n_letters, n_categories