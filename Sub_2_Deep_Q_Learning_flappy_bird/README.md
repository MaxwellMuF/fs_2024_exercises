# Submission:
This is the submission for the task "Exercise Deep Reinforcement Learning: Flappy Bird" of the lecture "Advanced Software Engineering (WiSe23)".

name: Sebastian von Rohrscheidt

matrikel: 102090

# Summary


# Setting:
Spider was used for training (but VS code for the repository).
1. The hyperparameters of the last training were as follows (Figure 1)![Screenshot (583)](https://github.com/MaxwellMuF/Submissions_Advanced_Software_Engineering_-WiSe23-/assets/148557718/f75dbfec-4c77-4708-b9b7-1f24ebce7d05)

2. The printouts and saves of the model were as follows (Figure 2). A current model was saved every 10k iterations in case the process crashes. A model was saved for every 100k for later analysis.![Screenshot (585)](https://github.com/MaxwellMuF/Submissions_Advanced_Software_Engineering_-WiSe23-/assets/148557718/6bf353a6-ec16-4a90-9a2e-3866d6f84a65)

3. The code was slightly modified for training and for optimizing the hyperparameters (Figure 3). The ideas implemented for this are explained in the next section.
![Screenshot (586)](https://github.com/MaxwellMuF/Submissions_Advanced_Software_Engineering_-WiSe23-/assets/148557718/8a8836ad-650a-4ee8-8852-ea072f7e1309)

# Training:
To get a better understanding of the flappy bird model, the agent was trained several times and with different hyperparameters:
1. flappy_bird_S-R_100_000: Default setting (given) of the hyperparameter. This agent is hardly successful, which is probably due to the too small number of iterations (or game rounds). In order to train a more successful agent, the hyperparameters were changed slightly.
2. flappy_bird_S-R_100_000_lr_e-5: Here the learning rate has been made slightly higher. This should allow the agent to learn the required actions more quickly, i.e. it should perform better with the same number of iterations. Unfortunately, this is still not enough to achieve a recognizably better result. In addition, overfitting is encouraged.
3. flappy_bird_S-R_200000_high_lr_eps: After a few more attempts, this is the last setting of hyperparameters. Three changes have been made:
    1. the learning rate with dacay. The agent starts with a high learning rate, which decreases over time (latest: lr*0.1 every 50% of data).
    2. epsilon: The rate of random actions with decay. The exploration rate was strongly increased at the beginning and then a damping (decrease) was built in with an e-function until the final_epsilon was reached at the end of the training.
    3. random_action: The probability of making a jump as a random action has been reduced (again with decay). Initially only 10% of random actions are a jump. With a further decay, the rate will be 50/50 again at the end (or for the last 30% of the training).

Thus, an agent could be trained that could pass 3-4 or more tunnels (after 200k training iterations, 1-2 tunnel after 100k iter), at least in some scenarios. However, it is suspected that this agent is overtrained and it remains to be seen whether he can ever master the perfect game and whether he needs less than 2m iterations of training to do so. The overtraining is visible when the agent (bird) hits a wall. This action seems to be far from the perfect game, even if it only occurs after 5-10 tunnels have been successfully passed. Nevertheless, this last agent (200k training iter) seems to perform better than the given 1m agent.

# Link to Git Repo
https://github.com/MaxwellMuF/Submissions_Advanced_Software_Engineering_-WiSe23-/tree/main/Sub_2_flappy_bird
