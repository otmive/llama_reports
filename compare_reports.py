HF_TOKEN="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"
import transformers
import torch


import fire
import re
import os
import torch
import csv

def get_score(output):
    match = re.search(r'Score:\s*(\d+)', output)
    if match:
        score = match.group(1)
    else:
        score = -1
    return score

def clean_report(report):
    # Remove the word "much obliged" if it appears
    cleaned_report = re.sub(r'(?i)much obliged\s*', '', report)

    # Remove "anonymized" and any surrounding special characters like ****
    cleaned_report = re.sub(r'\*+anonymized\*+', '', cleaned_report, flags=re.IGNORECASE)
    
    # Remove any leading or trailing spaces or newlines
    cleaned_report = cleaned_report.strip()

    cleaned_report = re.sub(r'[^\w\s]', '', cleaned_report).replace("\n"," ")

    return cleaned_report

def get_model(model):
    if model == 'biomistral':
        model_name = "BioMistral/BioMistral-7B"
    elif model == 'llama3':
        model_name = "../llama_reports/meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif model == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif model == "llama2":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    print(model_name)
    return model_name

def run_main(
    report_1: str,
    report_2: str,
    model: str="llama3",
    score: str="raw",
    output_file: str="output.txt",
    use_entities: bool=False,
    score_only: bool=False
    ):
    """
    Compare two reports and return a similarity score.
    """
    
    model_name = get_model(model)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        token=HF_TOKEN,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, 
        token=HF_TOKEN, 
    )

    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )    


    with open(report_1) as r:
        report_1 = r.read()
        report_1 = clean_report(report_1)
    with open(report_2) as r:
        report_2 = r.read()
        report_2 = clean_report(report_2)

            
        if use_entities:
            
            message = f"""
                Please provde a similarity score out of 10 for the two list of extracted entities from two radiology reports. Note which entities are 
                not present in both. Entities from report 1: {ents1}. Entities from report 2: {ents2}.
                Please give your output in the template
                Score:<score>, Reasoning:<reasoning>
            """
        else:
            message = f"""
                Please can you provide a similarity score out of 10 for these two reports.
                Focus on technical content rather than style or phrasing.
                Please give your output in the template
                Score:<score>, Reasoning:<reasoning>
                Report 1: {report_1}, Report 2: {report_2}
                """

        messages = [

            {
                "role": "user",
                "content": message,
            }
        ]
        # Print the assistant's response
        outputs = pipe(messages, max_new_tokens=512)
        print(outputs[0]['generated_text'][0]['content'])
        assistant_out = outputs[0]['generated_text'][1]['content']
        
        assistant_out = outputs[0]['generated_text'][1]['content']

        # Check if the file exists
        file_exists = os.path.exists(output_file)

        # Open the CSV file in append mode
        if not score_only:
            print(assistant_out)
            with open(output_file, "w") as f:
                f.write(assistant_out)
                f.close()
        else:
            score = get_score(assistant_out)
            print("Score: ", score)
            with open(output_file, "w") as f:
                f.write(score)
                f.close()
 

    


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()

