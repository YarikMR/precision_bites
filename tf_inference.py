import os

from transformers import pipeline
import argparse
import pandas as pd
from pathlib import Path
import json

"""
python tf_inference.py --file data/ISEAR/ISEAR.csv --model_path models/ISEAR --text_col text --output ISEAR_evaluation.csv --cuda 2
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF inference for tf_classifier.py')
    parser.add_argument('--file', type=str, required=True, help="csv or tsv file to classify")
    parser.add_argument("--model_path", type=str, required=True,
                        help="path to the model directory. It should have the model_info.json")
    parser.add_argument('--sep', type=str, default=',')
    parser.add_argument('--text_col', type=str, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cuda', default="1", type=str, help="GPU to use")

    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    data_file = pd.read_csv(args.file, sep=args.sep)
    model_path = Path(args.model_path)

    # load model info
    with open(model_path.joinpath('config.json')) as file:
        model_info = json.load(file)

    # load pipeline
    pipe = pipeline("text-classification", model = model_path, tokenizer = model_info['_name_or_path'])

    predictions = pipe(data_file[args.text_col].apply(str).to_list())
    data_file[model_path.name] = [pre['label'] for pre in predictions]
    data_file[f"{model_path.name}_score"] = [pre['score'] for pre in predictions]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data_file.to_csv(args.output, index=False)



