HF_TOKEN="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"
import transformers
import torch

from entities_vis import color_words, color_code
import fire
import sys
import os
import torch
import csv
import pandas as pd
import re
import spacy
import numpy as np

def clean_report(report):
    # Remove the word "much obliged" if it appears
    cleaned_report = re.sub(r'(?i)much obliged\s*', '', report)

    # Remove "anonymized" and any surrounding special characters like ****
    cleaned_report = re.sub(r'\*+anonymized\*+', '', cleaned_report, flags=re.IGNORECASE)
    
    # Remove any leading or trailing spaces or newlines
    cleaned_report = cleaned_report.strip()

    # replace ? with a space
    cleaned_report = re.sub(r'\?', ' ', cleaned_report)

    cleaned_report = re.sub(r'[^\w\s]', '', cleaned_report).replace("\n"," ")

    cleaned_report = cleaned_report.lower()

    return cleaned_report

def split_report(report):
    ind = False
    match_words = ["impression", "conclusion", "opinion"]
    report_list = report.split()
    back_list = report_list[::-1]
    for i, word in enumerate(back_list):
        for w in match_words:
            if w in word.lower():
                ind = len(report_list) - 1 - i
    if not ind:
        ind = round(0.75*len(report_list))
    find = report_list[:ind]
    imp = report_list[ind:]
    findings = " ".join(find)
    impression = " ".join(imp)
    
    return [findings, impression]
    

def lemmatize(note, nlp):
    doc = nlp(note)
    lemNote = [wd.lemma_ for wd in doc]
    return " ".join(lemNote)

def get_context(pipe, report1, report2, entity, num_times_seen):
    context_length = 20
    if len(report1) > context_length:
        # get all indexes of word in report
        instances = [i for i in range(len(report1)) if report1.startswith(entity, i)]
        # get index of word to build window around
        index = instances[num_times_seen]
        if index - context_length < 0:
            ind1 = 0
        else:
            ind1 = index-context_length
        if index + context_length > len(report1):
            ind2 = len(report1)
        else:
            ind2 = index + context_length
        report1 = report1[ind1:ind2]

    if len(report2) > context_length:
        # get all indexes of word in report
        instances = [i for i in range(len(report2)) if report2.startswith(entity, i)]
        index = instances[num_times_seen]
        if index - context_length < 0:
            ind1 = 0
        else:
            ind1 = index-context_length
        if index + context_length > len(report2):
            ind2 = len(report2)
        else:
            ind2 = index + context_length
        report2 = report2[ind1:ind2]

    if report1 == report2:
        return "same"
    else:

        message = f"""Can you say whether the entity: '{entity}' is used in the same context or different context in these two texts.
        Text1: '{report1}'
        Text2: '{report2}'
        Please reply with a single word answer, either 'same' or 'different'
        """

        messages = [

            {
                "role": "user",
                "content": message,
            }
        ]
        outputs = pipe(messages, max_new_tokens=512)
        out = outputs[0]['generated_text'][1]['content']
        return out

def get_pipe():

    model_name = "../llama_reports/meta-llama/Meta-Llama-3.1-8B-Instruct"


    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )

    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )
    return pipe


def split_multi_word_items(array):
    result = []
    for item in array:
        words = item.split()  # Split item by spaces
        result.extend(words)  # Add each word separately to the result list
    return result

def get_entities_for_block_pair(block1, block2):
    nlp = spacy.load("en_core_sci_lg")
    doc1 = nlp(block1)
    doc2 = nlp(block2)
    ents1 = [e.text for e in doc1.ents]
    ents2 = [e.text for e in doc2.ents]
    ents1 = split_multi_word_items(ents1)
    ents2 = split_multi_word_items(ents2)
    pipe = get_pipe()
    splits1 = block1.split()
    splits2 = block2.split()
    matched = [e for e in ents1 if e in ents2 and splits1.count(e) == splits2.count(e)]
    missing = [e for e in ents1 if e not in ents2 and splits1.count(e) > splits2.count(e)]
    surplus = [e for e in ents2 if e not in ents1 and splits1.count(e) < splits2.count(e)]
    mismatched = []
    all_ents = []
    for e in matched:
        context = get_context(pipe, block1, block2, e, all_ents.count(e))
        if "different" in context:
            mismatched.append(e)
            matched.remove(e)
        all_ents.append(e)

    return matched, mismatched, missing, surplus


def calculate_score(findings_entities, impression_entities, weights):

    def calculate_chunk(arr1, arr2,arr3,arr4):
        matches = len(arr1)
        mismatches = len(arr2)
        missing = len(arr3)
        surplus = len(arr4)
        w_missing = weights[1]
        w_surplus = weights[2]
        w_mismatch = weights[0]

        all = matches + missing*w_missing + surplus*w_surplus + mismatches*w_mismatch

        if all == 0:
            return 1
        else:
            return round(matches/all, 2)
    findings = calculate_chunk(findings_entities[0], findings_entities[1], findings_entities[2], findings_entities[3])
    impression = calculate_chunk(impression_entities[0], impression_entities[1], impression_entities[2], impression_entities[3])

    # weight impressions as twice as important
    score = round((findings + 3*impression)/4,2)

    return score

def compare_two_reports(report_1_file, report_2_file, w_mismatch, w_missing, w_surplus, plot_file=None):

    with open(report_1_file) as r:
        report_1 = r.read()

    with open(report_2_file) as r:
        report_2 = r.read()

    report_1 = clean_report(report_1)
    [findings1, impression1] = split_report(report_1)
    report_2 = clean_report(report_2)
    [findings2, impression2] = split_report(report_2)
        
    #block_pairs, missing, surplus = get_block_pairs(findings1, findings2)

    entities1 = get_entities_for_block_pair(findings1, findings2)
    entities2 = get_entities_for_block_pair(impression1, impression2)
    if plot_file:
        color_code(findings1, findings2, entities1, impression1, impression2, entities2, plot_file)

    print(calculate_score(entities1, entities2, [w_mismatch, w_missing, w_surplus]))
    return calculate_score(entities1, entities2, [w_mismatch, w_missing, w_surplus])

def run_main(
    report_1: str,
    report_2: str,
    w_mismatch: float=2,
    w_missing: float=1.5,
    w_surplus: float=1,
    plot_file: str=None
    ):
    """
    Compare two reports and return a score of similarity.
    """
    compare_two_reports(report_1, report_2, w_mismatch, w_missing, w_surplus, plot_file)

def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()

