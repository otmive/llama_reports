import numpy as np
import argparse
import spacy
import re

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

def mcse(nlp, entlist1, entlist2):
    C = 0
    entlist1_new = entlist1.copy()
    for e in entlist1:
        if e in entlist2:
            entlist2.remove(e)
            entlist1_new.remove(e)
            C += 1
    S_list = []
    for e in entlist1_new:
        sims = []
        for e2 in entlist2:
            sims.append(nlp(e).similarity(nlp(e2)))
        if np.mean(sims)>0:
            Si = max(sims)/(max(sims) + np.mean(sims))
        else:
            Si = 0
        S_list.append(Si)
    cosine_score = (C + sum(S_list))/len(entlist1)
    return cosine_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report1", type=str)
    parser.add_argument("report2", type=str)
    args = parser.parse_args()
    nlp = spacy.load("en_core_sci_lg")
    with open(args.report1) as r:
        report_1 = r.read()    
    with open(args.report2) as r:
        report_2 = r.read()

    report_1 = clean_report(report_1)
    report_2 = clean_report(report_2)
    doc1 = nlp(report_1)
    doc2 = nlp(report_2)
    ents1 = [e.text for e in doc1.ents]
    ents2 = [e.text for e in doc2.ents]
    score = mcse(nlp, ents1, ents2)
    print(score)
    return(score)




if __name__ == "__main__":
    main()
