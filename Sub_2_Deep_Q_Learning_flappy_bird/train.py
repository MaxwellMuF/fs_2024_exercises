"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import time
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

# Change the working directory to a new directory (replace with the path you want) # C:\Users\sebas\Documents\Data_Science\WS_23_24\Fortgeschrittene Softwaretechnik\Submissions_Advanced_Software_Engineering_WiSe23\Sub_2_flappy_bird
new_working_directory = "/workspace/Sub_2_Deep_Q_Learning_flappy_bird"
os.chdir(new_working_directory) # /workspace/Sub_2_Deep_Q_Learning_flappy_bird/train.py
print("get here")
# Get and print the new current working directory
new_current_working_directory = os.getcwd()
print(f"The new current working directory is: {new_current_working_directory}")

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.9)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_iters", type=int, default=2_000_000)
    parser.add_argument("--replay_memory_size", type=int, default=50_000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    save_print_at_iteration = 100_000
    time_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = []
    iter = 0
    epsilon_interval_end = -np.log(opt.final_epsilon / opt.initial_epsilon) # project or transform interval 
    epsilon_x_values = np.linspace(0, epsilon_interval_end, opt.num_iters) # of iteration on e-function
    
    lr_interval_end = -np.log(1e-6*0.1 / 1e-4) 
    lr_x_values = np.linspace(0, lr_interval_end, opt.num_iters)
    # multiply lr by 1e4 and make decay ever 25% if iteration two lines below
    # opt.lr = opt.lr * 1e3
    # print "Perform a random action" only 10% of the time (Iterations as well)
    count_rand_act = 0
    print(f"Training is runnning, iter_max: {opt.num_iters}, lr: {opt.lr}, gpu: {torch.cuda.is_available()}")
    while iter < opt.num_iters:
        #  make lr decay ever 25%
        if iter % (opt.num_iters//10) == 0: # int(opt.num_iters/2) == 0:
            #  opt.lr = opt.lr /10
            epsilon = opt.initial_epsilon*np.exp(-epsilon_x_values[iter])
            opt.lr = 1e-4 * np.exp(-lr_x_values[iter])

            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        prediction = model(state)[0]
        # Exploration or exploitation
        # epsilon = opt.initial_epsilon*np.exp(-epsilon_x_values[iter]) # Exponential decay of the exploration-rate
        u = random()
        random_action = u <= epsilon
        if random_action:
            count_rand_act += 1
            # if count_rand_act % save_print_at_iteration == 0:
            #     print("Perform a random action")
            # The following decay reduces the initially high probability of not jumping during a random action
            no_act_prob = 9 - int(iter*10 / opt.num_iters) # decay of no act prob [0] every 10% of iter
            no_act_prob = max(no_act_prob,1)
            # random action by sampling list of zeros and one one, e.g. [0,0,0,1]  
            action = sample([0]*no_act_prob + [1], 1)[0] # start with 90% [0] and 10% [1] end with 50% [0] and 50% [1]
                                                         # sequence of no_act_prob: [9,8,7,6,5,4,3,2,1,1]
        else:
            try:
                action = torch.argmax(prediction)[0]
            except:
                action = torch.argmax(prediction).item()

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        if iter % save_print_at_iteration == 0:
            train_progress = "\nIteration: {}/{}, Action: {}, no_act_prob: {}, Loss: {}, Epsilon: {}, lr: {} Reward: {}, Q-value: {}".format(
                iter,
                opt.num_iters,
                action,
                no_act_prob,
                loss,
                round(epsilon, 7),
                opt.lr,
                reward, 
                torch.max(prediction))
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction), iter)

        # Save model ever save point
        if iter % save_print_at_iteration == 0:
            time_iter = round(time.time() - time_start)
            torch.save(model, f"{opt.saved_path}/flappy_bird_SR_{iter}_dec_lr_eps_{time_iter//3600}h_{(time_iter%3600)//60}m")
            time_progress = f"Current time: {time_iter//3600}h {time_iter//60}m {time_iter%60}s"
            with open(f"{opt.saved_path}/train_prints_19_01_25_1.txt", "a") as file:
                file.write(f"{train_progress}\n{time_progress}\n\n")

    time_end = round(time.time() - time_start)
    print(f"End of training time: {time_end//3600}h {time_end//60}m {time_end%60}s")


if __name__ == "__main__":
    opt = get_args()
    train(opt)
