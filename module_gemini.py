import re
import google.generativeai as genai 
from spacy_help_functions import get_entities, create_entity_pairs

#We declare a list with entity of interests as shown below
entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]

#We make function called make call to api to create entity pairs for gemini
def make_call_to_api(sentence, r):
    #Sentence entity_pairs function here to create entity pairs
    sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
    #We do a for loop
    for ep in sentence_entity_pairs:
        # To minimize the use, use subject-object tuples with the relevant entities depending on r
        # If valid pair is found according to criteria, we return True. If not, we return False
        # Different scenario is listed below and it's clearly laid out
        if r==1 or r==2:
          if ep[1][1] == 'PERSON' and ep[2][1] == 'ORGANIZATION':
            return True # e1=Subject, e2=Object
          if ep[2][1] == 'PERSON' and ep[1][1] == 'ORGANIZATION':
            return True  # e1=Subject, e2=Object
        elif r==3:
          if ep[1][1] == 'PERSON' and (ep[2][1] == 'LOCATION' or ep[2][1] == 'CITY' or ep[2][1] == 'STATE_OR_PROVINCE' or ep[2][1] == 'COUNTRY'):
            return True  # e1=Subject, e2=Object
          if ep[2][1] == 'PERSON' and (ep[1][1] == 'LOCATION' or ep[1][1] == 'CITY' or ep[1][1] == 'STATE_OR_PROVINCE' or ep[1][1] == 'COUNTRY'):
            return True  # e1=Subject, e2=Object
        elif r==4:
          if ep[1][1] == 'ORGANIZATION' and ep[2][1] == 'PERSON':
            return True  # e1=Subject, e2=Object
          if ep[2][1] == 'ORGANIZATION' and ep[1][1] == 'PERSON':
            return True  # e1=Subject, e2=Object
    return False

def check_string_regex(s):
    # Pattern to match any of the specified words
    pattern = r'none|not specified|N\/A|Not Available|null|NULL|Not Mentioned|Not Applicable|NA|Not Provided|Subject: ,|Object: ,|Object: ]|PERSON\'S NAME|ORGANIZATION|LOCATION|CITY|STATE_OR_PROVINCE|COUNTRY|he|she|NOT FOUND|Subject: \|'
    # Search the string for any match
    return not re.search(pattern, s, re.IGNORECASE)

def get_prompt_text(q, relation_type, sentence):
    # Relation types to specific prompt formats, one can use 1,2,3,4
    # Refined prompts to do the relation. Used prompts as shown below for 1-4 relation types for specific prompt texts
    # Each template is designed to extract different relationships
    # Due to Gemini's inconsistency, we experiemented with several prompts to give best results compared to the transcript
    # Based on TA's advice, we experiemented the prompts that fit input/output and are most consistent and give best and specific results
    prompts = {
        1: f"Prompt: Given the sentence, extract all instances where person's name (subjects) and educational institutions (objects) are mentioned together."
    f"For objects or subjects, do not include companies like Google/Microsoft as they are not educational institution."
    f"Try to remove pronouns like he/she and ensure person’s name is human name."
    f"For each identified relationship: Ensure the subject is a person's name and confirm the object is a legitimate educational institution, identifiable by keywords such University, College, School, or Academy."
    f"Format each relationship found in the sentence as follows: Subject: [PERSON'S NAME] | Object: [ORGANIZATION]\nSentence: {sentence}",
        2: f"Prompt: Analyze the given sentence to identify and extract all instances that clearly indicate an employment "
    f"relationship, focusing on where a person's name (subject) is associated with an organization (object) they work for. "
    f"Clarify that subjects should be identifiable human names, avoiding any confusion with companies, subsidiaries, "
    f"or other non-individual entities. Objects should be legitimate organizations, recognizable through keywords "
    f"such as Corporation, Company, Foundation, or Inc. Exclude pronouns and ensure clarity in distinguishing between "
    f"subjects and objects.\n\n"
    f"For each relationship found, format as follows: "
    f"Subject: [PERSON'S NAME] | Object: [ORGANIZATION]\n"
    f"Ensure that the subject refers exclusively to individuals and the object to the organizations they work for.\n"
    f"Sentence: {sentence}",
        3: f"Analyze the given sentence to identify and extract all instances that "
    "clearly indicate a 'live-in' relationship, focusing on where a person's name (subject) "
    "is associated with a geographic location (object) they live in. "
    "Clarify that subjects should be identifiable human names, avoiding any confusion with "
    "companies, subsidiaries, or other non-individual entities. Objects should be legitimate "
    "geographic locations, recognizable through keywords such as city, state, country, or region. "
    "Exclude pronouns and ensure clarity in distinguishing between subjects and objects.\n\n"
    "For each relationship found, format as follows: "
    "Subject: [PERSON'S NAME] | Object: [GEOGRAPHIC LOCATION]\n"
    "Ensure that the subject refers exclusively to individuals and the object to the geographic "
    "locations where they live.\n"
    f"Sentence: {sentence}",
    4: f"""
Extract company-key figure relationships from sentences. Identify when a company (subject) and a top member employee or key figure (object) are mentioned together. 

Criteria:
1. Subjects are legitimate companies or organizations, excluding educational institutions and non-corporate entities.
2. Objects are individuals in significant roles (e.g., executives, founders).
3. Exclude references without specific names, notable roles, or that are ambiguous.
4. Format: "Subject: [COMPANY NAME] | Object: [INDIVIDUAL NAME]".

Example: Given "Jeff Bezos, the founder of Amazon, introduced new policies," the output should be "Subject: Amazon | Object: Jeff Bezos".

Implementation should validate:
- The subject as a valid company or organization.
- The object as a notable individual associated with the subject.

Avoid false positives and ensure clear association between subject and object. Given sentence: {sentence}
"""
    }
    #We retrieve and return the relation 
    return prompts.get(relation_type, "Invalid relation type.")

# Function to get content generation from the Gemini API
def get_gemini_completion(prompt_text, api_key, model_name='gemini-pro', max_tokens=100, temperature=0.1, top_p=1, top_k=32):
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
        # We return the response text
        return response.text
    except Exception as e:
        print(f"\tError during Gemini completion: {str(e)}")
        return ""
