import spacy
import os
import re
import argparse

def clean_report(report):
    # Remove the word "much obliged" if it appears
    cleaned_report = re.sub(r'(?i)much obliged\s*', '', report)

    # Remove "anonymized" and any surrounding special characters like ****
    cleaned_report = re.sub(r'\*+anonymized\*+', '', cleaned_report, flags=re.IGNORECASE)
    
    # Remove any leading or trailing spaces or newlines
    cleaned_report = cleaned_report.strip()

    cleaned_report = re.sub(r'[^\w\s]', '', cleaned_report).replace("\n"," ")

    return cleaned_report

def main():
    parser = argparse.ArgumentParser(description="Process radiology reports to extract entities.")
    parser.add_argument("report_input", type=str, help="Input file or directory of reports.")
    parser.add_argument("--output", type=str, default="entities_output", help="Output directory or file for entities.")
    
    args = parser.parse_args()
    nlp = spacy.load("en_core_sci_md")
    if os.path.isdir(args.report_input):
        # create output folder
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        # loop through input folder 
        for file in os.listdir(args.report_input):
            entities = []
            with open(f"{args.report_input}/{file}") as r:
                report = r.read()
                report = clean_report(report)
            doc = nlp(report)
            for e in doc.ents:
                entities.append(e.text)
            string = ",".join(entities)
            print(string)
            with open(f"{args.output}/{file}", "w") as r:
                r.write(string)
                r.close()
           
    elif os.path.isfile(args.report_input):
        # return entities for single report
        entities = []
        print(args.report_input)
        with open(f"{args.report_input}") as r:
            report = r.read()
            report = clean_report(report)
        doc = nlp(report)
        for e in doc.ents:
            entities.append(e.text)
        string = ",".join(entities)
        if not args.output.endswith(".txt"):
            output = args.output + ".txt"
        else:
            output = args.output
        with open(output, "w") as r:
            r.write(string)
            r.close()
    
if __name__ == "__main__":
    main()