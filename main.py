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
import google.generativeai as genai 
import module_spanbert as module_spanbert
import re


def check_string_regex(s):
    # Pattern to match any of the specified words
    pattern = r'none|not specified|N\/A'
    # Search the string for any match
    return not re.search(pattern, s, re.IGNORECASE)

def get_prompt_text(relation_type, sentence):
    # Relation types to specific prompt formats, one can use 1,2,3,4
    # TODO: We may need to get this more refined
    prompts = {
        1: f"Given a sentence, extract all names of persons (subjects) and schools attended (objects). Extract only when subject and object are there. Don't extract pronouns unless it's declared in sentence. [Subject: PERSON'S NAME, Object: ORGANIZATION]\nSentence: {sentence}",
        2: f"Given a sentence, extract all names of persons (subjects) and organizations they work for (objects). Extract only when subject and object are there. Output: [Subject: PERSON'S NAME, Object: ORGANIZATION]\nSentence: {sentence}",
        3: f"Given a sentence, extract all names of persons (subjects) and their living locations (objects). Extract only when subject and object are there. Output: [Subject: PERSON'S NAME, Object: LOCATION]\nSentence: {sentence}",
        4: f"Given a sentence, extract all organizations (subjects) and top member employees (objects). Extract only when subject and object are there. Output: [Subject: ORGANIZATION, Object: PERSON'S NAME]\nSentence: {sentence}"
    }
    return prompts.get(relation_type, "Invalid relation type.")

# Function to get content generation from the Gemini API
def get_gemini_completion(prompt_text, api_key, model_name='gemini-pro', max_tokens=100, temperature=0.2, top_p=1, top_k=32):
    # print("\tProcessing sentence for extraction ...")
    # Configure Gemini API with the provided API key
    genai.configure(api_key=api_key)
    
    # Initialize Gemini model
    model = genai.GenerativeModel(model_name)
    # Set up configuration with parameters by following reference
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    
    try:
        # Create response based on the prompt text and configuration
        response = model.generate_content(prompt_text, generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"\tError during Gemini completion: {str(e)}")
        return ""

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
    documents.append(i['formattedUrl'])
  return documents

def get_plain_text(url):
    # Send a GET request to the URL
    print("\tFetching text from url ...")
    response = requests.get(url)

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
        print(f"=========== Iteration: {iteration+1} - Query: {q} ===========\n")
        target_relations = {}
        spanbert = SpanBERT("./pretrained_spanbert")
        for i, url in enumerate(items):
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
                if(j!= 0 and j % 5 == 0):
                    print(f"\tProcessed {j} / {num_of_sentences} sentences")
                if model == "-spanbert":
                    res, n = module_spanbert.spanbert_process(spanbert, t, r, sentence)
                    print("Dict res", res)
                    extracted = len(res)
                    if n != 0:
                        sen_counter += 1
                        rel_counter += n
                else:
                    print("Gemini")
                    all_relations = set()
                    if len(all_relations) < k:  # Only process new relations if below k
                    # print(f"\tProcessing sentence: {sentence}")
                        prompt_text = get_prompt_text(r, sentence)
                        response_text = get_gemini_completion(prompt_text, gemini_api_key)
                        # print(response_text)

                        # Accumulate only if new and unique and going up to k relations(would need to clarify)
                        if "Subject" in response_text and "Object" in response_text and response_text.strip() not in all_relations:
                            if check_string_regex(response_text.strip()):
                                print(f"\t=== Extracted Relation ===")
                                print(f"\tSentence: {sentence}")
                                print(f"\tExtraction: {response_text.strip()}")
                                print("\t==========\n")
                                all_relations.add(response_text.strip())
                        else:
                            print(f"\tNo valid relation extracted from this sentence or duplicate found.\n")

            print(f"Extracted annotations for  {sen_counter}  out of total  {num_of_sentences}  sentences")
            print(f"Relations extracted from this website: {extracted} (Overall: {rel_counter})")
            print(f"\n================== ALL RELATIONS for relation type {r} ( {len(all_relations)} ) =================")
        for relation in all_relations:
            print(relation)
        iteration += 1
        break

if __name__ == "__main__":
    main()