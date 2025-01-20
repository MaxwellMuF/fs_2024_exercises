# Submission:
This is the submission for the task "Sub_7_Sentiment_Analyzer_BERT" of the lecture "Advanced Software Engineering (WiSe24/25)".

name: Sebastian von Rohrscheidt

matrikel: 102090

# Summary
The BERT model was trained on a v100 GPU on the BHT cluster. Small changes were made to the code for e.g. dependency issues and training documentation. The model was successfully trained and predicted the labels of the test prompts with 100% accuracy.

# Changes with respect to given code:
First Error:
- ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`
- Solution: replace `transformers[torch]` with `transformers[torch]`
Second Error:
- ImportError: cannot import name 'load_metric' from 'datasets'
- Solution: add to requirements `evaluate`, i.e. replace `dataset.load_metric` with  `evaluate.load` (see: https://discuss.huggingface.co/t/cant-import-load-metric-from-datasets/107524)

# Link to Git Repo
https://github.com/MaxwellMuF/fs_2024_exercises