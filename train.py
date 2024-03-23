import string
import random

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

from model import RNN 
from data_preprocessing import get_data  

path = "./data/names/*.txt"
letters = string.ascii_letters + " .,;'"
category_dict = get_data(path, letters)
categories = list(category_dict.keys())
n_letters = len(letters)
n_categories = len(categories)
hidden_dim = 128

def category_from_output(output):
    _, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return categories[category_i], category_i

def name_to_tensor(name: str) -> torch.Tensor:
    tensor = torch.zeros(len(name), 1, n_letters)
    for i, char in enumerate(name):
        tensor[i, 0, letters.index(char)] = 1
    return tensor

def random_training_pair():
    category = random.choice(categories)
    name = random.choice(category_dict[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    name_tensor = name_to_tensor(name)
    return category, name, category_tensor, name_tensor

def main():
    model = RNN(input_dim=n_letters, hidden_dim=hidden_dim, output_dim=n_categories)
        
    loss_fn = nn.CrossEntropyLoss()
    lr = 5e-3  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 100000

    model.to(device)
    losses = []
    epoch_bar = tqdm.tqdm(desc="Training", total=epochs, position=0)

    for epoch in range(1, epochs+1):  
        category, name, category_tensor, name_tensor = random_training_pair()
        category_tensor, name_tensor = category_tensor.to(device), name_tensor.to(device)
        
        hidden = model.init_hidden().to(device)
        
        for i in range(name_tensor.shape[0]):
            output, hidden = model(name_tensor[i], hidden)
        loss = loss_fn(F.softmax(output, dim=1), category_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 5000 == 0:
            guess, _ = category_from_output(output)
            verdict = "✔" if guess == category else "❌"
            print(f"\nEpoch: {epoch} | Loss: ({loss:.4f}) | Name: {name} | Guess: {guess} | Correct: {category} | Verdict: {verdict}")

        epoch_bar.update()

if __name__ == "__main__":
    main()
