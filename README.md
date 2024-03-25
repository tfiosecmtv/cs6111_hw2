# E6111 Project 2 
This is our README of our project 2. We achieve k tuples within the reasonable amount of iteration and we see the completed results within equal or less than the iterations the reference project takes for both SpanBERT and Gemini. We have commented the code with detail with explanation for each functions and commands. We have described our projects of how to run, explained clear descriptions of our project, and wrote out clear description of our query modification method below.

## Authors
Richard Han(dh3062)

Aidana Imangozhina(ai2523)

## Files we're submitting
project2.py

module_gemini.py

module_spanbert.py

transcript_gemini.pdf

transcript_spanbert.pdf

## Getting Started

### Usage/How to run

There are a few ways to run our project. First method is running locally and second is running on our server. Both were tested extensively.

If you want to run it in local machine, run with following
```
python3 project2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>
```

For instance, 
```
python3 project2.py -gemini AIzaSyBCa0Wc70pbXZGhPDAw6PN7TDzgJEqR-Gw 24edf1bcc7aaa4d1b AIzaSyDMSxypDwhP7LWrttowtJux164zWYWHhIc 3 0.7 "megan rapinoe redding" 2
```
```
or python3 project2.py -spanbert AIzaSyBCa0Wc70pbXZGhPDAw6PN7TDzgJEqR-Gw 24edf1bcc7aaa4d1b AIzaSyDMSxypDwhP7LWrttowtJux164zWYWHhIc 3 0.7 "megan rapinoe redding" 2
```

Commands to install necessary software if running on local machine:

If using python3 (These are the instructions used for VM, but we also used in our local machine). Execute the commands line by line in your terminal:
```
pip3 install --upgrade google-api-python-client
pip3 install beautifulsoup4
pip3 install -U pip setuptools wheel
pip3 install -U spacy
python3 -m spacy download en_core_web_lg
git clone https://github.com/larakaracasu/SpanBERT
cd SpanBERT
pip3 install -r requirements.txt
bash download_finetuned.sh
pip install -q -U google-generativeai
```

To run our project after setting up all the instructions, put project2.py, module_gemini.py, module_spanbert.py under SpanBERT folder since we encountered path issues when it was trying to access './pretrained_model'. After that execute the command from 'Usage/How to run' part from SpanBERT directory. After moving the files to SpanBERT folder and when you execute 'ls SpanBERT' outside of the Spanbert directory, you should get:

```
(base) MacBook-Pro-421:cs6111_hw2 user$ ls SpanBERT
LICENSE                 download_finetuned.sh   pretrained_spanbert     requirements.txt        temp.py
README.md               example_relations.py    pytorch_pretrained_bert spacy_help_functions.py
__pycache__             gemini_prompt.py        relations.txt           spanbert.py
```


### Keys / Engine ID

Google API key = AIzaSyBCa0Wc70pbXZGhPDAw6PN7TDzgJEqR-Gw 

Google Engine ID = 24edf1bcc7aaa4d1b

Gemini API key = AIzaSyDMSxypDwhP7LWrttowtJux164zWYWHhIc (you might need to generate new one since we encountered out of quota error due to frequent testing)

Threshold = 0.9 #This value can be changed by user and this is the common threshold value we used..

```
[-spanbert|-gemini]: either -spanbert or -gemini, to indicate which relation extraction method we are requestin
Google API Key: Google Custom Search Engine JSON API Key.
Google Engine ID: Google Custom Search Engine ID.
Gemini API key: Google Gemini API key.
<r> : is an integer between 1 and 4, indicating the relation to extract: 1 is for Schools_Attended, 2 is for Work_For, 3 is for Live_In, and 4 is for Top_Member_Employees
Precision: The target confidence value for SpanBERT model.
Query: The search query from user in double quotation.
<k>: an integer greater than 0, indicating the number of tuples that we request in the output
```

### Clear description of the internal design of our project

Input validation: After receiving user input, we first do the range validation for r, t, k using is_integer_between(value, min_value, max_value), is_real_number_between(value, min_value, max_value), is_positive_integer(value) functions. 

Search Google Function: We have the function get_documents(service, query, cx) which searches google with api key, cse_id, and search query parameter to search and return top 10 results utilizing Google Custom Search JSON API.

Declare relations set: We use dictionary if the model is SpanBERT since we need to keep track of the confidence for each tuple. In case of Gemini, we only use set since we need to avoid the duplicates.

Iteration process: We keep iteration process until we reach the target size of k for the relation set or dictionary. In the reference transcript for gemini model, if the iteration is more than 1 and the desired size of the relation set is achieved it does not traverse through other URLs. We also implemented that logic. However, we process all the URLs in the first iteration even if we achieve the target size k.

We query Google Custom Search Engine to obtain the URLs for the top-10 webpages by utilizing requests and Beautiful for fetching web pages and extracting text.

We utilize spacy for text annotation and entity recognition to understand the structure and entities that are existing. We also utilize SpanBERT and Gemini to analyze deeper for entity relationship. Sentences are analyzed for entity pairs and extraction for both SpanBERT and Gemini.

We use relationship of interest and map them to specific entity pairs through our custom logic that we created for SpanBERT and Gemini models. For each sentence, entity pairs are identified based on the relevance of context based on desired relationship. We also did this for Gemini model as well. Further, we use Gemini API with genai.GenerativeModel for obtaining further entity relationship. We extract data with additional contextual information after we utilize the prompt. We also utilize regex based checks to ensure cleanness of ouput and to present relevant output. We also followed specifications accordingly to create a input that can be used on CLI to follow seamlessly. 

## Detailed description for processing the URL content

Context extraction and processing: For each URL, we extract the content of the URL using get_plain_text(url) function where we only use first 10000 characters if the length of the documents is greater than 10000. We replace all new lines with whitespaces. When it's not possible to retrieve the content of the web page, we return None object. After the extraction of raw text from the web page, we process the web page using Spacy library to process the raw_text and extract sentences.

Minimizing API calls: For both SpanBERT and Gemini, we used the helper functions - get_entities, create_entity_pairs from spacy_help_functions provided in the project description. We check if the entity pairs have valid pairs based on the relation r to make sure we make less calls to SpanBERT and Gemini.

SpanBERT sentence processing: We create entity candidate pairs based on the result of create_entity_pairs. If entity pairs have desired entities based on relation r, we add it to candidate pairs. After extracting all candidate pairs for sentence, we make a call to the SpanBERT model and put the entity pair tuples to dictionary where value is the confidence. As specified in the project description, if there are duplicates, we only keep the one with the higher confidence value. If the confidence value is less than threshold, we ignore the tuple.

Gemini sentence processing: We fetch the raw HTMl content of given URL and extract text from it. Text is truncated to first 10,000 characters to manage processing load and focus on content.

We annotate the fetched plain text using spacy. Spacy NLP library is utilized. Model is applied to text producing doc object that contains token, entities and annotations.

We prepare relationship extraction by utilizing function make call to api and get prompt text. We determine whether the sentence contains entity pairs relevant to the needed relationship type and prepare a prompt for the gemini model. We analyze the entity pairs that match the specified criteria per specification indicated on relation type r. We check if entities fit the type(for instance Person with Organization) and confirm the relationship context.  If valid pair is found, we utilize prompt using get prompt text to alert Gemini model to find the specific relationship between identified entities. 

We further utilize Gemini API with API keys and model to find the relationship. We utilized the sample gemini code provided and specified parameters like max output token, temperature, top_k, and top_p to control text generation process. With our input, Gemini is able to find a relationship based on our prompt constructed. 

Counters: We declare counters before processing each sentence to keep track of how many relations are extracted per sentence and how many of them were included in the relations set. Also, we keep track of how many sentence we processed.

Printing the relation set: We also print the relations set after each iteration, so the user can see what type of relations were extracted per each iteration. If the desired size of k is achieved we stop the iteration and print the entire set. For SpanBERT model, we indicated after top-k tuples that the desired size of k is achieved, but we print all the tuples for user to compare in the reverse sorted order by confidence. For Gemini, we print all the tuples in the relation set without ordering since we do not have confidence value.

Generating new query: We keep track of used queries using a dictionary where the value is a query string and key is a boolean value. For SpanBERT model, we iterate through tuples in the reverse sorted order based on their confidence and check if the query was used before using the dictionary. If it was used before, we skip it. Otherwise, we add it to dictionary and make a new round of calls in a new iteration. Same goes for Gemini, but we arbitrarily pick the tuples since we have no confidence value and track their usage history using the dictionary.

Final print: After printing all relations after each iteration, if the target size of k is reached, we stop the program and print the total number of iterations.

