# Submission:
This is the submission for the task "Exercise Reinforcement Learning: Taxi Driver" of the lecture "Advanced Software Engineering (WiSe24/25)".

name: Sebastian von Rohrscheidt

matrikel: 102090

# Comment:
The AI was trained and tested as specified.

# Changes with respect to given code:
First Error:
- ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`
- Solution: replace `transformers[torch]` with `transformers[torch]`
Second Error:
- ImportError: cannot import name 'load_metric' from 'datasets'
- Solution: add to requirements `evaluate`, i.e. replace `dataset.load_metric` with  `evaluate.load` (see: https://discuss.huggingface.co/t/cant-import-load-metric-from-datasets/107524)

# Link to Git Repo
https://github.com/MaxwellMuF/Submissions_Advanced_Software_Engineering_-WiSe23-/tree/main/Sub_1_Taxi_Driver