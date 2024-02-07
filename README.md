# Affective Text Generation Across Different Bite Precision Values

## Overview
This project explores the generation of affective text across various quantization precisions in Large Language Models (LLMs). It includes scripts for training a classifier to assess affective content, evaluating text outputs using the classifier, and generating affective text under different precision constraints.

Author: Yarik Menchaca Resendiz

Email: st176412@stud.uni-stuttgart.de

## Project Structure
- `train_classifier.py`: Script for training the classifier on affective text data.
- `tf_inference.py`: Script for evaluating generated text using the pre-trained classifier.
- `text_generation.py`: Script for generating affective text with specified bite precision values.
- `env.yml`: Conda environment file to set up the necessary dependencies.

## Getting Started

### Prerequisites
To run the scripts, you will need to set up a Python environment with the required dependencies. The easiest way to do this is by using Conda, a package and environment management system.

### Setup Environment
1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if it's not already installed.
2. Navigate to the project directory in the terminal.
3. Create a Conda environment using the `environment.yml` file provided:
   ```sh
   conda env create -f env.yml
   ```
### Data Acquisition
The ISEAR dataset is required to train the classifier. Please download the dataset from the original source as follows:

Visit the ISEA dataset repository.
Follow the instructions provided to download the dataset.

### Training the Classifier
To train the classifier on the ISEAR dataset
```sh
python train_classifeir.py --file Path_to_ISEAR.csv --model_path models/ISEAR --text_col text --output output_path.csv --cuda 2
```
### Generating Affective Text
To generate affective text with a specific bite precision value:

```sh
python text_generation.py
```

### Evaluating Generated Text
To evaluate the affective content of generated text using the classifier:

```sh
python tf_inference.py --file Path_to_generated_text.csv --model_path models/ISEAR --text_col text --output output_path.csv --cuda 2
```

# License

This project is licensed under the MIT License - see the LICENSE file for details.