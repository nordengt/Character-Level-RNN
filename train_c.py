import argparse

import string
import random

import torch
from torch import nn

from tqdm import tqdm

from data_preprocessing import get_data
from model import RNN, T_RNN
from utils import output_category, plot_losses, name_to_tensor

path = "./data/names/*.txt"
letters = string.ascii_letters + " .,;'"

category_name_dict, categories, n_letters, n_categories = get_data(path, letters)
n_hidden = 128

def random_choice(cat_list: list) -> str:
    return cat_list[random.randint(0, len(cat_list)-1)]

def random_sample():
    category = random_choice(categories)
    name = random_choice(category_name_dict[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    name_tensor = name_to_tensor(name, letters, n_letters)
    return name_tensor, category_tensor, name, category

def main(model_type: str = "scratch") -> None:
    if model_type == "pytorch":
        model = T_RNN(n_letters, n_hidden, n_categories)
    else:
        model = RNN(n_letters, n_hidden, n_categories)
    
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.005
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations = 100_000

    losses = []
    current_loss = 0

    model.to(device)
    model.train()
    for iteration in tqdm(range(1, iterations+1)):
        name_tensor, category_tensor, name, category = random_sample()
        
        hidden = model.init_hidden().to(device)
        name_tensor, category_tensor = name_tensor.to(device), category_tensor.to(device)

        for i in range(name_tensor.shape[0]):
            output, hidden = model(name_tensor[i], hidden)

        loss = loss_fn(output, category_tensor)
        current_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 5000 == 0:
            out_cat = output_category(output, categories)
            result = "Correct" if out_cat == category else "Incorrect"
            print(f"\n Name: {name.ljust(20)} Actual: {category.ljust(20)} Predicted: {out_cat.ljust(20)} Result: {result}")
        
        if iteration % 1000 == 0:
            losses.append(current_loss / 1000)
            current_loss = 0

    torch.save(model.state_dict(), f"./saved/{model_type}_model.pth")
    plot_losses(losses, f"./results/{model_type}-classification.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("--model_type", type=str, default="scratch")
    args = parser.parse_args()
    main(args.model_type)