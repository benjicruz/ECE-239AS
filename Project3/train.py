"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb
from tqdm import tqdm

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig

def solver(model_name):
    # Initialize the model
    if model_name == "bigram":
        config = BigramConfig
        model = BigramLanguageModel(config)
    elif model_name == "minigpt":
        config = MiniGPTConfig
        model = MiniGPT(config)
    else:
        raise ValueError("Invalid model name")
    
    # Load the dataset
    train_dataset = TinyStoriesDataset(
        config.path_to_data,
        mode="train",
        context_length=config.context_length,
    )
    eval_dataset = TinyStoriesDataset(
        config.path_to_data, mode="test", context_length=config.context_length
    )

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.batch_size, pin_memory=True
    )
    # print the size of each dataloader
    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Eval dataloader size: {len(eval_dataloader)}")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print number of parameters in the model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))

    # Initialize wandb if you want to use it
    if config.to_log:
        wandb.init(project="dl2_proj3")

    # Create the save path if it does not exist
    if not Path.exists(config.save_path):
        Path.mkdir(config.save_path, parents=True, exist_ok=True)

    ### ==================== START OF YOUR CODE ==================== ###
    """
    You are required to implement the training loop for the model.

    The code below is a skeleton for the training loop, for your reference. 
    You can fill in the missing parts or completely set it up from scratch.

    Please keep the following in mind:
    - You will need to define an appropriate loss function for the model.
    - You will need to define an optimizer for the model.
    - You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
    - It is recommended that you save the model weights every `config.save_iterations` iterations. You can also just save the model with the best training loss.

    NOTE : 
    - Please check the config file to see the different configurations you can set for the model.
    - The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
    not a required part of the assignment. 
    - Feel free to experiment with the parameters and I would be happy to talk to you about them if interested.
    """

    ### ========= TODO : START ========= ###
    # Define the loss function
    loss = nn.CrossEntropyLoss()
    

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ### ======== TODO : END ========= ###

    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2
        )

    model.train()
    model.to(device)

    for i, (context, target) in enumerate(train_dataloader):

        train_loss = None # You can use this variable to store the training loss for the current iteration
        ### ======== TODO : START ========= ###
        # Do the forward pass, compute the loss, do the backward pass, and update the weights with the optimizer.
        context, target = context.to(device), target.to(device)
        logits = model(context)
        
        logits = rearrange(logits, "b t v -> (b t) v")
        target = rearrange(target, "b t -> (b t)")
        
        train_loss = loss(logits, target)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        ### ======== TODO : END ========= ###

        if config.scheduler:
            scheduler.step()

        del context, target # Clear memory

        if i % config.log_interval == 0:
            print(
                f"Iteration {i}, Train Loss: {train_loss}",
            )

            if config.to_log:
                wandb.log(
                    {
                        "Train Loss": train_loss
                    },
                    step=i
                )
            
        if i % config.eval_log_interval == 0:
            model.eval()
            eval_loss = None # You can use this variable to store the evaluation loss for the current iteration
            ### ======== TODO : START ========= ###
            # Compute the evaluation loss on the eval dataset.
            
            total_eval_loss = 0
            num_batches = 0
            for j, (eval_context, eval_target) in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader)//20, desc="Evaluating")):
                eval_context, eval_target = eval_context.to(device), eval_target.to(device)
                eval_logits = model(eval_context)

                eval_logits = eval_logits.view(-1, eval_logits.size(-1))
                eval_target = eval_target.view(-1)

                loss_value = loss(eval_logits, eval_target)
                total_eval_loss += loss_value.item()
                num_batches += 1
                # only use 1/10 of the eval length and the break out of the loop
                if j > len(eval_dataloader) // 20:
                    break
                
            eval_loss = total_eval_loss / num_batches
            
            print(
                f"\033[94mIteration {i}, Eval Loss: {eval_loss}\033[0m",
            )
            if config.to_log:
                wandb.log(
                    {
                        "Eval Loss": eval_loss,
                    },
                    step=i
                )
            
            ### ======== TODO : END ========= ###
            


            model.train()

        # Save the model every config.save_iterations
        if i % config.save_iterations == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "iteration": i,
                },
                config.save_path / f"mini_model_checkpoint_{i}.pt",
            )

        if i > config.max_iter:
            break
        # if the loss is sufficiently low, break out of the loop
        if train_loss < 3.0 and eval_loss < 3.5:
            print("Loss is sufficiently low, stopping training.")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "iteration": i,
                },
                config.save_path / f"mini_model_sufficient_loss_checkpoint_{i}.pt",
            )
            break
                