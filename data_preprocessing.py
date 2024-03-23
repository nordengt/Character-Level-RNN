import glob
import unicodedata
from typing import List, Dict

def unicode_to_ascii(name: str, letters: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", name)
        if not unicodedata.combining(c) and c in letters
    )

def get_category_names(filename: str, letters: str, encoding: str = "utf-8") -> List[str]:
    names = open(filename, encoding=encoding).read().strip().split("\n")
    return [unicode_to_ascii(name, letters) for name in names]

def get_data(path: str, letters: str) -> Dict[str, List[str]]:
    files = glob.glob(path)
    category_dict = {}
    for filename in files:
        category = filename.split("\\")[-1].split('.')[0]
        category_dict[category] = get_category_names(filename, letters)
    return category_dict