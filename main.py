from bs4 import BeautifulSoup
import spacy
from spanbert import SpanBERT
from googleapiclient.discovery import build
import sys
import requests
import module_spanbert as module_spanbert
import module_gemini as module_gemini
from collections import defaultdict
import re

predicates = {
    1: "Schools_Attended",
    2: "Work_For",
    3: "Live_In",
    4: "Top_Member_Employees"
}

predicates_bert = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}

def spacy_process(raw_text):
    print("\tAnnotating the webpage using spacy...")
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(raw_text)
    return doc

def get_documents(service, query, cx):
  res = (
        service.cse()
        .list(
            q=query,
            cx=cx,
        )
        .execute()
    )
  documents = []
  for i in res['items']:
    documents.append(i['link'])
  return documents

def get_plain_text(url):
    # Send a GET request to the URL
    print("\tFetching text from url ...")
    response = requests.get(url, timeout=5)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the plain text from the parsed HTML
        # plain_text = soup.get_text(strip=True)
        text = ' '.join(soup.stripped_strings)
        plain_text = ' '.join(text.split())
        # Truncate the text to its first 10,000 characters if it exceeds that length
        if len(plain_text) > 10000:
            plain_text = plain_text[:10000]

        return plain_text
    else:
        # If the request was not successful, print an error message
        print(f"Error: Unable to retrieve content from {url}. Status code: {response.status_code}")
        return None

# Input validation methods

def is_integer_between(value, min_value, max_value):
    try:
        int_value = int(value)
        return min_value <= int_value <= max_value
    except ValueError:
        return False

def is_real_number_between(value, min_value, max_value):
    try:
        float_value = float(value)
        return min_value <= float_value <= max_value
    except ValueError:
        return False

def is_positive_integer(value):
    return is_integer_between(value, 1, float('inf'))

def main():
    usage_msg = "Usage: python3 main.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>"
    if len(sys.argv) < 8:
        #This is the format of using the program. If less than the correct arg, we exit
        print(usage_msg)
        sys.exit(1)
    _, model, api_key, cse_id, gemini_api_key, r, t, q, k = sys.argv

    t = float(t)
    r = int(r)
    k = int(k)

    # Validation
    if model not in ['-spanbert', '-gemini']:
        sys.exit("Model must be '-spanbert' or '-gemini'.")

    if not is_integer_between(r, 1, 4):
        sys.exit("<r> must be an integer between 1 and 4.")

    if model == '-spanbert' and not is_real_number_between(t, 0, 1):
        sys.exit("<t> must be a real number between 0 and 1 for -spanbert.")

    if not is_positive_integer(k):
        sys.exit("<k> must be a positive integer.")

    print('____')
    print(f"Parameters:")
    print(f"Client key\t= {api_key}")
    print(f"Engine key\t= {cse_id}")
    print(f"Gemini key\t= {gemini_api_key}")
    print(f"Method\t\t= {model}")
    print(f"Relation\t= {predicates[r]}")
    print(f"Threshold\t= {t}")
    print(f"Query\t\t= {q}")
    print(f"# of Tuples\t= {k}")
    print(f"Loading necessary libraries; This should take a minute or so ...")

    service = build(
        "customsearch", "v1", developerKey=api_key
    )
    iteration = 1
    res = defaultdict(int)
    all_relations = set()
    used_q = {q: True}
    while len(res) < k and len(all_relations) < k:
        items = get_documents(service, q, cse_id)
        if not items:
            print("No results found.")
            sys.exit(1)
            continue
        print(f"=========== Iteration: {iteration} - Query: {q} ===========\n")
        if model == "-spanbert":
            spanbert = SpanBERT("./pretrained_spanbert")
        for i, url in enumerate(items):
            print("LEN RES:", len(res))
            print(f"URL ( {i+1} / 10): {url}")
            raw_text = get_plain_text(url)
            if raw_text == None:
                continue
            print(f"\tWebpage length (num characters): {len(raw_text)}")
            docs = spacy_process(raw_text)
            num_of_sentences = 0
            for j, sentence in enumerate(docs.sents):
                num_of_sentences += 1
            print(f"\tExtracted {num_of_sentences} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
            sen_counter = 0
            rel_counter = 0
            extracted = 0
            
            for j, sentence in enumerate(docs.sents):
                if(j != 0 and j % 5 == 0):
                    print(f"\tProcessed {j} / {num_of_sentences} sentences")
                if model == "-spanbert":
                    sc, rc, ec = module_spanbert.spanbert_process(spanbert, t, r, sentence, res)
                    sen_counter += sc
                    rel_counter += rc
                    extracted += ec
                else:
                    prompt_text = module_gemini.get_prompt_text(q, r, sentence)
                    response_text = module_gemini.get_gemini_completion(prompt_text, gemini_api_key)
                    if "Subject" in response_text and "Object" in response_text and response_text.strip() not in all_relations:
                        response_text_lines = response_text.strip().split("\n")
                        sen_counter += 1
                        for line in response_text_lines:
                        # Check if both "Subject" and "Object" are present in the line
                            if "Subject" in line and "Object" in line:
                                # If the line meets the criteria and it's not already in the list of all relations
                                # Assuming all_relations is a list of strings, each representing a previously processed relation
                                if line not in all_relations:
                                    # Add the line to the list of extracted relations
                                    if module_gemini.check_string_regex(line.strip()):
                                        print(f"\t=== Extracted Relation ===")
                                        print(f"\tSentence: {sentence}")
                                        print(f"\tExtraction: {line.strip()}")
                                        print("\t==========\n")
                                        extracted += 1
                                        rel_counter += 1
                                        all_relations.add(line.strip())
                        
                    # else:
                    #     print(f"\tNo valid relation extracted from this sentence or duplicate found.\n")

            print(f"Extracted annotations for  {sen_counter}  out of total  {num_of_sentences}  sentences")
            print(f"Relations extracted from this website: {extracted} (Overall: {rel_counter})")
        
        if model == "-spanbert":
            sorted_items = sorted(res.items(), key=lambda x: x[1], reverse=True)
            for key, value in sorted_items:
                new_q = key[0] + " " + key[2]
                if new_q in used_q:
                    continue
                else:
                    used_q[new_q] = True
                    q = new_q
                    break
        else:
            for relation in all_relations:
                matches = re.findall(r'(?:subject|object):\s*(.*?)(?:\s*;|$)', relation, re.IGNORECASE)
                # Join the extracted strings into a new string
                new_str = ' '.join(matches)
                if new_str in used_q:
                    continue
                else:
                    used_q[new_str] = True
                    q = new_str
                    break
        iteration += 1
    if model == '-spanbert':
        print(f"================== ALL RELATIONS for {predicates_bert[r]} ( {k} ) =================")
        sorted_items = sorted(res.items(), key=lambda x: x[1], reverse=True)
        num = 0
        for key, value in sorted_items:
            if num == k:
                break
            print(f"Confidence: {value} 		| Subject: {key[0]} 		| Object: {key[2]}")
            num += 1
        
    else:
        print(f"================== ALL RELATIONS for {predicates[r]} ( {len(all_relations)} ) =================")
        for rel in all_relations:
            print(rel)
    print(f"Total # of iterations = {iteration-1}")

if __name__ == "__main__":
    main()