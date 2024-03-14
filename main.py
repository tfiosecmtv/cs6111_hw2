import pprint
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from bs4 import BeautifulSoup
import spacy
from spanbert import SpanBERT
from googleapiclient.discovery import build
import sys
import requests
import module_spanbert

def spacy_process(raw_text):
    print("          Annotating the webpage using spacy...")
    nlp = spacy.load("en_core_web_sm")
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
    documents.append(i['formattedUrl'])
  return documents

def get_plain_text(url):
    # Send a GET request to the URL
    print("          Fetching text from url ...")
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the plain text from the parsed HTML
        plain_text = soup.get_text()

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
    print(f"Relation\t= {r}")
    print(f"Threshold\t= {t}")
    print(f"Query\t\t= {q}")
    print(f"# of Tuples\t= {k}")
    print(f"Loading necessary libraries; This should take a minute or so ...")

    service = build(
        "customsearch", "v1", developerKey=api_key
    )
    # currently doing for spanbert
    iteration = 0
    while True:
        items = get_documents(service, q, cse_id)
        if not items:
            print("No results found.")
            sys.exit(1)
            continue
        print(f"=========== Iteration: {iteration} - Query: {q} ===========")
        target_relations = {}

        for i, url in enumerate(items):
            print(f"URL ( {i} / 10): {url}")
            raw_text = get_plain_text(url)
            print(f"          Webpage length (num characters): {len(raw_text)}")
            docs = spacy_process(raw_text)
            num_of_sentences = len(docs)
            print(f"Extracted {num_of_sentences} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
            sen_counter = 0
            rel_counter = 0
            extracted = 0
            for j, sentence in enumerate(docs):
                if(j % 5 == 0):
                    print(f"          Processed {j+1} / {num_of_sentences} sentences")
                if model == "'-spanbert":
                    res, n = module_spanbert.spanbert_process(target_relations, t, r, sentence)
                    extracted = len(res)
                    if n != 0:
                        sen_counter += 1
                        rel_counter += n
            print(f"Extracted annotations for  {sen_counter}  out of total  {num_of_sentences}  sentences")
            print(f"Relations extracted from this website: {extracted} (Overall: {rel_counter}")

        iteration += 1
        break

if __name__ == "__main__":
    main()