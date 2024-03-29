import argparse
import string

import torch
from model import G_RNN

from utils import name_to_tensor, output_category, category_to_tensor

letters = string.ascii_letters + " .,;'"
categories = [
    "Arabic",
    "Chinese",
    "Czech",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Irish",
    "Italian",
    "Japanese",
    "Korean",
    "Polish",
    "Portuguese",
    "Russian",
    "Scottish",
    "Spanish",
    "Vietnamese"
]

n_letters = len(letters)
n_hidden = 128
n_categories = len(categories)

def main(country: str, start: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = G_RNN(n_letters, n_categories, n_hidden, n_letters)
    model.load_state_dict(torch.load(f"./saved/generation_model.pth", map_location=device))
    model.to(device)

    model.eval()
    generated_name = start
    with torch.inference_mode():
        category_tensor = category_to_tensor(country, categories, n_categories).to(device)
        name_tensor = name_to_tensor(start, letters, n_letters).to(device)
        hidden = model.init_hidden().to(device)
        for _ in range(20):
            output, hidden = model(category_tensor, name_tensor[0], hidden)
            pred = torch.argmax(output, dim=1).item()
            if pred == n_letters-1: break
            else:
                letter = letters[pred]
                generated_name += letter
            name_tensor = name_to_tensor(letter, letters, n_letters).to(device)

    print(f"The generated {country} name is {generated_name}")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation Script")
    parser.add_argument("--country", type=str, help="Country of name origin")
    parser.add_argument("--start", type=str, help="Starting word of the name")
    args = parser.parse_args()
    main(args.country, args.start)