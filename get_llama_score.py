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

def compare(
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
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

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
        return assistant_out
 

    


def main():
    fire.Fire(compare)


if __name__ == "__main__":
    main()

