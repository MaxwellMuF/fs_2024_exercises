# Submission:
This is the submission for the task "Sub_2_Deep_Q_Learning_flappy_bird" of the lecture "Advanced Software Engineering (WiSe24/25)".

name: Sebastian von Rohrscheidt

matrikel: 102090

# Summary
The most time was invested in this task. The goal of beating the given benchmark models was achieved. This was accomplished by increasing the initialization of the lr and epsion and decreasing them over time using an e function. In addition, the probability of a jump was also first set to 10% and ended up at 50% over time (default). Please note the training prints for 2025 (Sub_2_Deep_Q_Learning_flappy_bird\trained_models\train_prints_19_01_25.txt).

Further work could show that individual hyperparameters contributed more or less to the success. But this would require many more experiments and a metric would have to be programmed, e.g. like '100 test runs end on average after x minutes with a failed attempt (collision with obstacle). However, it is assumed that a decay through an e function (also beyond fbird) can lead to good results. This concept is well known in damped harmonic oscillators (approach to solving the DGL) and is used in mechanics, electrodynamics and quantum mechanics. It is reasonable to apply this to 'simulated neurons' or similar.

# Setting:
Spider was used for training (but VS code for the repository).
1. The hyperparameters of the last training were as follows (Figure 1)![Screenshot (583)](https://github.com/user-attachments/assets/aa123b22-a290-4448-82d1-21ffff8dba5e)


2. The printouts and saves of the model were as follows (Figure 2). A current model was saved every 10k iterations in case the process crashes. A model was saved for every 100k for later analysis.![Screenshot (585)](https://github.com/user-attachments/assets/97fe0243-2e85-4dfa-bd8a-a616a36d85db)


3. The code was slightly modified for training and for optimizing the hyperparameters (Figure 3). The ideas implemented for this are explained in the next section.
![Screenshot (586)](https://github.com/user-attachments/assets/65099dc9-2cf4-40a8-9eb3-7d7b9f494ff5)


# Training:
To get a better understanding of the flappy bird model, the agent was trained several times and with different hyperparameters:
1. flappy_bird_S-R_100_000: Default setting (given) of the hyperparameter. This agent is hardly successful, which is probably due to the too small number of iterations (or game rounds). In order to train a more successful agent, the hyperparameters were changed slightly.
2. flappy_bird_S-R_200000_high_lr_eps: After a few more attempts, this is the last setting of hyperparameters. Three changes have been made:
    1. the learning rate with dacay. The agent starts with a high learning rate, which decreases over time (latest: lr*0.1 every 50% of data).
    2. epsilon: The rate of random actions with decay. The exploration rate was strongly increased at the beginning and then a damping (decrease) was built in with an e-function until the final_epsilon was reached at the end of the training.
    3. random_action: The probability of making a jump as a random action has been reduced (again with decay). Initially only 10% of random actions are a jump. With a further decay, the rate will be 50/50 again at the end (or for the last 30% of the training).
3. flappy_bird_SR_2000000_dec_lr_eps_18h_22m: This model was trained on a GPU in 2025. It seems to be as good or better than the given 2m benchmark model. However, the "flappy_bird_SR_1300000_dec_lr_eps_11h_56m" model seems to be more interesting, as already with this number of training steps no more failed attempts could be observed. All 2025 trained models have the same settings and hyperparameters as in the current code version.

# Conclusion
The models "flappy_bird_S-R_xxxx_high_lr_eps" were trained one year ago on a laptop and are surprisingly good (probably because the lr decay is completed at 200k, while with the 200k model (trained 2025) a very high lr leads to temporary overfitting).
The "flappy_bird_SR_xxxx_dec_lr_eps_xh_xm" models were trained in 2025 on a v100 GPU on the BHt cluster. Nevertheless, it took 18h for the 2m model. On a CPU it would have taken more than twice as long.
However, it is noticeable that the 1m model (2025) is significantly better than the given 1m benchmark model. Furthermore, no more failed attempts could be observed from the 1.3m model onwards. However, this is not meaningful, as mentioned in the summary, a metric would be needed for an appropriate evaluation. For example, a collision could be observed by chance with the given 2m model. The same could also apply to the 2025 models.

# Link to Git Repo
https://github.com/MaxwellMuF/fs_2024_exercises