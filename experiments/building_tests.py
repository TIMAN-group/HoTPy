import argparse
import sys
import os
import re
import json
#os.environ['HF_HOME'] = Path to HF Home

import xml.etree.ElementTree as ET

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

sys.path.append('..')
from src.HoT import HoT

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
            'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
            'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
            'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
            'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
            'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
            'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
            "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def get_llama():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device_map)

    return model, tokenizer

def extract_ents(model, tokenizer, text):
    messages = [
    {"role": "system", "content": "Summarize the following passage by creating a machine readable comma seperated list of things that the passage is about. Only output the machine readable list and nothing else."},
    {"role": "user", "content": text},
    ]
    with torch.no_grad():
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def split_and_clean_string(s):
    parts = re.split(r'[\t,]| {2,}|\n', s)

    cleaned_parts = [part.strip() for part in parts if part.strip() and part not in stop_words]
    
    return cleaned_parts

def split_text_dynamic(txt, num_parts):
    words = txt.split()
    avg_chunk_size = len(words) // num_parts
    chunks = [' '.join(words[i:i + avg_chunk_size]) for i in range(0, len(words), avg_chunk_size)]
    if len(chunks) > num_parts:
        # Merge the last two chunks if we have more than desired parts
        chunks[-2] += ' ' + chunks[-1]
        chunks = chunks[:-1]
    return chunks

def process_text(model, tokenizer, txt, max_retries=3):
    """Process text, retrying with smaller chunks if OOM occurs."""
    retries = 0
    while retries <= max_retries:
        num_parts = 2 ** retries  # Exponential backoff
        try:
            chunks = split_text_dynamic(txt, num_parts)
            all_results = []
            for chunk in chunks:
                try:
                    chunk_result = extract_ents(model, tokenizer, chunk)
                    term_list = split_and_clean_string(chunk_result)
                    all_results.extend(term_list)
                except torch.cuda.OutOfMemoryError:
                    print("CUDA out of memory while processing chunk. Skipping this chunk.")
                    torch.cuda.empty_cache()
                    continue

            return all_results
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory. Retrying with {num_parts * 2} parts.")
            torch.cuda.empty_cache()
            retries += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    print("Max retries reached. Skipping this text.")
    return []  # Return an empty result if all retries fail

def llama_build(file_path):
    model, tokenizer = get_llama()

    files = [f for f in os.listdir(file_path) if f.endswith('.json')]
    
    files.sort(key=lambda x: os.path.splitext(x)[0])

    graph = HoT()
    hyperedges = {}
    
    for filename in tqdm(files, desc="Processing files"):  # Outer progress bar
        filepath = os.path.join(file_path, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)  # Load JSON content
                for name, doc in tqdm(data.items(), desc=f"Processing data in {filename}", leave=False):  # Inner progress bar
                    node_id = graph.add_node(doc)
                    for sec in re.split(r'\n{2,}', doc):
                        term_list = process_text(model, tokenizer, sec)
                        for keyword in term_list:
                            if keyword.isalpha():
                                lower_kw = keyword.lower()
                                hyperedges.setdefault(lower_kw, set())
                                hyperedges[lower_kw].add(node_id)


            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {e}")

    for kw, node_set in hyperedges.items():
        graph.add_hyperedge(node_set, kw)

    graph.to_xml(file_path+"HoT/llama_graph.xml")

def bert_build(file_path):
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=sentence_model)

    files = [f for f in os.listdir(file_path) if f.endswith('.json')]
    
    files.sort(key=lambda x: os.path.splitext(x)[0])

    graph = HoT()
    hyperedges = {}
    
    for filename in tqdm(files, desc="Processing files"):  # Outer progress bar
        filepath = os.path.join(file_path, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)  # Load JSON content
                for name, doc in tqdm(data.items(), desc=f"Processing data in {filename}", leave=False):  # Inner progress bar
                    node_id = graph.add_node(doc)
                    for w in re.split(r'\n{2,}', doc):
                        f = kw_model.extract_keywords(w, keyphrase_ngram_range=(1, 2), use_maxsum=True)
                        for keyword, _ in f:
                            if keyword.isalpha():
                                lower_kw = keyword.lower()
                                hyperedges.setdefault(lower_kw, set())
                                hyperedges[lower_kw].add(node_id)


            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {e}")

    for kw, node_set in hyperedges.items():
        graph.add_hyperedge(node_set, kw)

    graph.to_xml(file_path+"HoT/bert_graph.xml")

    
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("mode", choices=["bert_build", "llama_build", "tfidf_build"], help="Mode: build or run")
    parser.add_argument("filepath", type=str, help="Path to the file")

    args = parser.parse_args()

    if args.mode == "bert_build":
        bert_build(args.filepath)
    elif args.mode == "llama_build":
        llama_build(args.filepath)
    else:
        print(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
