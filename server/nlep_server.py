import openai
import google.generativeai as palm

import csv

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import os
from urllib.parse import urlparse


app = FastAPI()

fs_prompt = open('prompts.txt').read()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    instruction: str
    api_key_str: str
    exist_code: str
    platform: str
    model: str


def parse_api_key(api_key_str):
    print(api_key_str)
    if 'api_key =' in api_key_str:
        return api_key_str.split('api_key =')[1].replace('\'', '').strip()
    else:
        return api_key_str


def construct_fspy_prompt(fs_prompt, inst_str, input_txt = 'None', exist_code = 'none'):
    if exist_code == 'none':
        prompt = f'''{fs_prompt}\n\n### Instruction: {inst_str}\n### Input: {input_txt}\n### Python program:'''
    else:
        prompt = f'''{fs_prompt}\n\n### Instruction: {inst_str}\n### Input: {input_txt}\n### Python program:\n```\n{exist_code.strip()}'''
    return prompt


def gpt4_py(ques_str, api_key_str, exist_code, platform, model='gpt-4'):
    api_key = parse_api_key(api_key_str)

    prompt = construct_fspy_prompt(fs_prompt, ques_str, exist_code = exist_code)
    
    if platform == 'gpt':
        openai.api_key = api_key

        gpt4_output = openai.ChatCompletion.create(
            model = model,
            messages = [{'role': 'user', 'content': prompt}],
            temperature = 0.5,
            top_p = 1.0,
            max_tokens = 1024
        )
        gen_txt = gpt4_output['choices'][0]['message']['content'].replace('```python', '```')
    
    elif platform == 'palm':
        palm.configure(api_key = api_key)
        completion = palm.generate_text(
            model = f'models/{model}',
            prompt = prompt,
            temperature = 0.5,
            # The maximum length of the response
            max_output_tokens = 1024,
        )
        gen_txt = completion.result
    
    else:
        return 'Platform not supported.'

    if exist_code != 'none' and gen_txt.startswith('```'):
        gen_txt = gen_txt[3:]

    if exist_code != 'none':
        gen_txt = f'\n{gen_txt}'

    res_str = (
        prompt + gen_txt
    ).split('### Python program:')[-1].strip()

    sections = res_str.split('```')
    if len(sections) > 1:
        ans_str = sections[1].strip()
    else:
        ans_str = sections[0].strip()
    
    return ans_str


@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    # Do something with the file
    content = await file.read()
    content_list = content.decode('utf-8').split('\n')[:3]
    content_str = '\n# '.join(content_list)
    prompt = f"# First three rows of the input file:\n# {content_str}\n# ...\nfile_path = 'cache/{file.filename}' # Please fill\ninput_file = open(file_path)"
    # For demonstration purposes, we'll just return the filename
    return {"filename": file.filename, "message": f"{prompt}"}



@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    ans_str = gpt4_py(
        item.instruction, item.api_key_str, item.exist_code,
        item.platform, item.model
    )

    return {"output": ans_str, "item_id": item_id}