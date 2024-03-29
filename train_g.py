import string
import random

import torch
from torch import nn

from tqdm import tqdm

from data_preprocessing import get_data
from model import G_RNN
from utils import plot_losses, name_to_tensor, category_to_tensor, target_to_tensor

path = "./data/names/*.txt"
letters = string.ascii_letters + " .,;'"

category_name_dict, categories, n_letters, n_categories = get_data(path, letters)
n_hidden = 128

def random_choice(cat_list: list) -> str:
    return cat_list[random.randint(0, len(cat_list)-1)]

def random_sample():
    category = random_choice(categories)
    name = random_choice(category_name_dict[category])
    category_tensor = category_to_tensor(category, categories, n_categories)
    name_tensor = name_to_tensor(name, letters, n_letters)
    target_tensor = target_to_tensor(name, letters, n_letters)
    return name_tensor, category_tensor, target_tensor

def main():
    model = G_RNN(n_letters, n_categories, n_hidden, n_letters)
    
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.0005
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations = 100_000

    losses = []
    current_loss = 0

    model.to(device)
    model.train()

    pbar = tqdm(total=iterations, desc="Loss: 0.00")

    for iteration in range(1, iterations+1):
        name_tensor, category_tensor, target_tensor = random_sample()
        
        hidden = model.init_hidden().to(device)
        name_tensor = name_tensor.to(device)
        category_tensor = category_tensor.to(device)
        target_tensor = target_tensor.unsqueeze(-1).to(device)

        loss = 0
        for i in range(name_tensor.shape[0]):
            output, hidden = model(category_tensor, name_tensor[i], hidden)
            l = loss_fn(output, target_tensor[i])
            loss += l 

        current_loss += loss.item() / name_tensor.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 1000 == 0:
            losses.append(current_loss / 1000)
            current_loss = 0
            pbar.set_description(f"Loss: {losses[-1]:.2f}")
        pbar.update(1)

    pbar.close()

    torch.save(model.state_dict(), f"./saved/generation_model.pth")
    plot_losses(losses, f"./results/generation.png")

if __name__ == "__main__":
    main()