import argparse
import string

import torch
from model import RNN, T_RNN

from utils import name_to_tensor, output_category

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

def main(name: str, model_type: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = T_RNN(n_letters, n_hidden, n_categories) if model_type == "pytorch" else RNN(n_letters, n_hidden, n_categories)

    model.load_state_dict(torch.load(f"./saved/{model_type}_model.pth", map_location=device))
    model.to(device)

    model.eval()
    with torch.inference_mode():
        name_tensor = name_to_tensor(name, letters, n_letters)
        name_tensor = name_tensor.to(device)
        hidden = model.init_hidden().to(device)
        for i in range(name_tensor.shape[0]):
            output, hidden = model(name_tensor[i], hidden)
        
    category = output_category(output, categories)
    print(f"The name {name} is: {category}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--name", type=str, required=True, help="Name to predict category for")
    parser.add_argument("--model_type", type=str, default="scratch", help="Type of model (scratch or pytorch)")
    args = parser.parse_args()
    main(args.name, args.model_type)