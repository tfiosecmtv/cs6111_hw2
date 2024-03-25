import argparse  
import requests 
import spacy  
from bs4 import BeautifulSoup  
import google.generativeai as genai  

# Initialize spaCy model for processing
print("Loading necessary libraries; This should take a minute or so ...")
nlp = spacy.load("en_core_web_sm")

# Function to fetch and process content
def fetch_webpage_content(url):
    print(f"\tFetching text from url: {url} ...")
    try:
        # Send HTTP GET request to the URL
        response = requests.get(url)
        # Raise HTTPError if the response status is returning an error
        response.raise_for_status()
        # Parse webpage content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text content from all <p> tags, join them, and limit to 10000 characters
        # text_content = ' '.join(p.text for p in soup.find_all('p'))[:10000]
        text_content = soup.get_text()
        text_content = text_content[:10000]
        print(f"\tWebpage length (num characters): {len(text_content)}")
        # Process text content with spaCy to split it into sentences
        doc = nlp(text_content)
        # Declare sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        # Follow format in the reference for printing
        print(f"\tAnnotating the webpage using spacy...")
        print(f"\tExtracted {len(sentences)} sentences.")
        return sentences
    except Exception as e:
        print(f"\tError fetching {url}: {str(e)}")
        return []

# Function to retrieve search results using Google Custom Search API
def fetch_search_results(query, api_key, cse_id, num_results=10):
    print(f"Fetching top {num_results} search results for query: '{query}'")
    # Request URL and parameters for Google Custom Search
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id,
        'num': num_results
    }
    urls = []
    try:
        # Send HTTP GET request with the search query and parameters
        response = requests.get(search_url, params=params)
        # Raise HTTPError when needed
        response.raise_for_status()
        # Parse search results and extract URLs
        search_results = response.json()
        # Process urls
        urls = [item['link'] for item in search_results.get('items', [])]
        print(f"Fetched {len(urls)} URLs.")
    except Exception as e:
        print(f"Error fetching search results: {str(e)}")
    return urls

# Function to generate prompt text for the Gemini API based on given relation type and sentence
def get_prompt_text(relation_type, sentence):
    # Relation types to specific prompt formats, one can use 1,2,3,4
    # TODO: We may need to get this more refined
    prompts = {
        1: f"Given a sentence, extract all names of persons (subjects) and schools attended (objects). Extract only when subject and object are there. Don't extract pronouns unless it's declared in sentence. ",
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
# Main function
def main(args):
    print("____")
    print("Parameters:")
    print(f"Client key\t= {args.google_api_key}")
    print(f"Engine key\t= {args.google_engine_id}")
    print(f"Gemini key\t= {args.gemini_api_key}")
    print(f"Method\t\t= {args.method}")
    print(f"Relation\t= {args.r}")
    print(f"Threshold\t= {args.t}")
    print(f"Query\t\t= {args.q}")
    print(f"# of Tuples\t= {args.k}")
    print(f"=========== Iteration: 0 - Query: {args.q} ===========\n")

    all_relations = set()
    urls = fetch_search_results(args.q, args.google_api_key, args.google_engine_id, 10)

    for i, url in enumerate(urls, 1):
        print(f"\nURL ({i} / {len(urls)}): {url}")
        sentences = fetch_webpage_content(url)
        num_sentences = len(sentences)
        for j, sentence in enumerate(sentences):
            if(j % 5 == 0):
                print(f"Processed {j} / {num_sentences} sentences")
            if len(all_relations) < args.k:  # Only process new relations if below k
                # print(f"\tProcessing sentence: {sentence}")
                prompt_text = get_prompt_text(args.r, sentence)
                response_text = get_gemini_completion(prompt_text, args.gemini_api_key)
                # print(response_text)

                # Accumulate only if new and unique and going up to k relations(would need to clarify)
                if "Subject" in response_text and "Object" in response_text and response_text.strip() not in all_relations:
                    print(f"\t=== Extracted Relation ===")
                    print(f"\tSentence: {sentence}")
                    print(f"\tExtraction: {response_text.strip()}")
                    print("\t==========\n")
                    all_relations.add(response_text.strip())
                # else:
                    # print(f"\tNo valid relation extracted from this sentence or duplicate found.\n")
            # else:
                # Continue processing without accumulating, providing completeness of parsing
                # print(f"\tSkipped sentence processing after reaching {args.k} unique relations.")

    # Output all unique relations
    print(f"\n================== ALL RELATIONS for relation type {args.r} ( {len(all_relations)} ) =================")
    for relation in all_relations:
        print(relation)

# Entry point with all arg 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract relations using Gemini API.")
    parser.add_argument("-m", "--method", choices=['spanbert', 'gemini'], required=True)
    parser.add_argument("google_api_key")
    parser.add_argument("google_engine_id")
    parser.add_argument("gemini_api_key")
    parser.add_argument("r", type=int, choices=range(1, 5))
    parser.add_argument("t", type=float)
    parser.add_argument("q")
    parser.add_argument("k", type=int)
    
    args = parser.parse_args()
    main(args)
