import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import argparse
from pathlib import Path
import numpy as np


"""
#Isear Emotions
python standard_hgf.py --file data/ISEAR/ISEAR.csv --text_col text --class_col class --output models/ISEAR

# enVent Emotions
python standard_hgf.py --file data/envent/train_crowd-enVent_generation_appraisal_bin.tsv --text_col generated_text --class_col emotion --output models/enVent-emotion 

"""

def encode_class(list_clases):
    # Encode labels using LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(list_clases)
    class_id = label_encoder.transform(list_clases)

    # Create label dictionary mapping integer labels to string labels
    id2label = {i: label_encoder.classes_[i] for i in range(len(label_encoder.classes_))}

    # Create inverse label dictionary mapping string labels to integer labels
    label2id = {v: k for k, v in id2label.items()}

    return class_id, label2id, id2label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torch classifier')
    parser.add_argument('--file', type=str, required=True, help="csv or tsv")
    parser.add_argument('--sep', type=str, default='\t')
    parser.add_argument('--text_col', type=str, required=True)
    parser.add_argument('--class_col', type=str, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cuda', default="1", type=str, help="GPU to use")
    parser.add_argument('--base_model', default="roberta-base", type=str)
    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data from CSV file
    dataset = pd.read_csv(args.file, sep=args.sep)

    # Encode labels
    dataset['encoded_label'], label2id, id2label = encode_class(dataset[args.class_col].apply(str).to_list())

    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(dataset[args.text_col],
                                                                        dataset['encoded_label'],
                                                                        test_size=0.2, random_state=42)


    # Load pre-trained tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model)
    model = RobertaForSequenceClassification.from_pretrained(args.base_model,
                                                             num_labels=len(label2id),
                                                             label2id = label2id,
                                                             id2label=id2label)

    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(train_labels.tolist())
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(val_encodings['input_ids']),
        torch.tensor(val_encodings['attention_mask']),
        torch.tensor(val_labels.tolist())
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=output_path.joinpath('logs'),
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=50,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model='eval_loss'
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([item[0] for item in data]),
            'attention_mask': torch.stack([item[1] for item in data]),
            'labels': torch.stack([item[2] for item in data])
        }
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    evaluation = trainer.evaluate()

    model.save_pretrained(output_path)
    #np.savetxt('eval_results.txt', evaluation)

