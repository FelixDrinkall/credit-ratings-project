# Model Training and Evaluation

This repository contains the code for training and evaluating a machine learning model to predict financial outcomes based on various features, including text data.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Arguments](#arguments)
- [Class Descriptions](#class-descriptions)
- [Functions](#functions)

## Requirements

To install the required libraries, run:

```bash
pip install -r requirements.txt

## Usage

To run the script, use:

```bash
python main.py --lag <lag_value> --feature_type <feature_type_value> --no_text <no_text_flag> --no_text_features <no_text_features_value> --only_text <only_text_flag>

For example:

```bash
python main.py --lag 0 --feature_type 'nrc_lex' --no_text False --no_text_features 'all' --only_text False

## Code Structure

The main script is structured as follows:

- **Imports:** Necessary libraries and modules.
- **Argument Parsing:** Parsing command-line arguments.
- **Helper Functions:** Additional functions for text processing and data formatting.
- **ModelTrainer Class:** The core class for training, evaluating, and saving the model.

## Arguments

The script accepts the following command-line arguments:

- `--lag`: Integer, default is 0. Specifies the quarters of lag.
- `--feature_type`: String, default is 'nrc_lex'. Specifies the method for encoding text.
- `--no_text`: Boolean, default is False. Indicates if no text features should be included.
- `--no_text_features`: String, default is 'all'. Specifies the type of features to exclude when `--no_text` is True.
- `--only_text`: Boolean, default is False. Indicates if only text features should be included.
