# Model Training and Evaluation

This repository contains the code for training and evaluating a machine learning model to predict financial outcomes based on various features, including text data. More processing code will be uploaded in due course.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Arguments](#arguments)
- [Citation](#citation)

## Requirements

To install the required libraries, run:

```bash
pip install -r requirements.txt
```

## Usage

To run the script, use:

```bash
python main.py --no_text <no_text_flag> --no_text_features <no_text_features_value> --only_text <only_text_flag>
```

For example:

```bash
python main.py  --no_text False --no_text_features 'all' --only_text False
```

## Code Structure

The main script is structured as follows:

- **Imports:** Necessary libraries and modules.
- **Argument Parsing:** Parsing command-line arguments.
- **Helper Functions:** Additional functions for text processing and data formatting.
- **ModelTrainer Class:** The core class for training, evaluating, and saving the model.

## Arguments

The script accepts the following command-line arguments:

- `--no_text`: Boolean, default is False. Indicates if no text features should be included.
- `--no_text_features`: String, default is 'all'. Specifies the type of features to exclude when `--no_text` is True.
- `--only_text`: Boolean, default is False. Indicates if only text features should be included.

## Citation 

If you use the code in this repository, please cite the following work:

@misc{drinkall2024traditionalmethodsoutperformgenerative,
      title={Traditional Methods Outperform Generative LLMs at Forecasting Credit Ratings}, 
      author={Felix Drinkall and Janet B. Pierrehumbert and Stefan Zohren},
      year={2024},
      eprint={2407.17624},
      archivePrefix={arXiv},
      primaryClass={q-fin.RM},
      url={https://arxiv.org/abs/2407.17624}, 
}
