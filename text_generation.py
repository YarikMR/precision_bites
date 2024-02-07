import pandas as pd
from tqdm import tqdm

seed_prompts = pd.read_csv('fundation_models/seed_prompts.csv')
emotion_isear = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']

# extend seed_prompts with classes
extend_seeds = list()
for index, row in seed_prompts.iterrows():
    for emotion in emotion_isear:
        s_prompt = row.prompt.replace('<class>', emotion)
        extend_seeds.append([row.prompt_id, s_prompt])
extend_seeds = pd.DataFrame.from_records(extend_seeds, columns=['prompt_id', 'prompt'])

import transformers
import os
from torch import bfloat16
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

model_id = 'meta-llama/Llama-2-13b-chat-hf'
model_config = transformers.AutoConfig.from_pretrained(
    model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    #load_in_8bit=True,
    #torch_dtype = torch.float16,
    device_map="auto"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

pipe = transformers.pipeline(model=model,
                             tokenizer=tokenizer,
                             task='text-generation',
                             top_k = 10,
                             top_p = 0.9,
                             temperature = 0.9,
                             do_sample = True,
                             repetition_penalty = 1,
                             #early_stopping=True,
#                                       no_repeat_ngram_size=2,
                             max_length=100)

# Generate text
generated_text = list()
for index, row in tqdm(extend_seeds.iterrows()):
    result = pipe(row.prompt)
    #print( result['generated_text'])
    text = result[0]['generated_text']
    generated_text.append([row.prompt_id, row.prompt, text])

df_generated_text = pd.DataFrame.from_records(generated_text, columns=['prompt_id', 'prompt', 'generated_text'])
df_generated_text.to_csv('fundation_models/13b_default.csv')