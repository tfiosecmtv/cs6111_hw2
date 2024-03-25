import re
import google.generativeai as genai 


def check_string_regex(s):
    # Pattern to match any of the specified words
    pattern = r'none|not specified|N\/A|Not Available|null|NULL|Not Applicable|NA|Not Provided|Subject: ,|Object: ,|Object: ]|PERSON\'S NAME|ORGANIZATION|LOCATION|CITY|STATE_OR_PROVINCE|COUNTRY|he|she|NOT FOUND'
    # Search the string for any match
    return not re.search(pattern, s, re.IGNORECASE)

def get_prompt_text(q, relation_type, sentence):
    # Relation types to specific prompt formats, one can use 1,2,3,4
    # TODO: We may need to get this more refined
    prompts = {
        1: f"Given the sentence, extract all instances where person's name (subjects) and educational institutions (objects) are mentioned together. Try to remove pronouns like he/she and ensure person’s name is human name. For each identified relationship: Ensure the subject is a person's name and confirm the object is a legitimate educational institution, identifiable by keywords such University, College, School, or Academy. Format each relationship found in the sentence as follows: Subject: [PERSON'S NAME] | Object: [ORGANIZATION]\nSentence: {sentence}",
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
        4:  f"Given the sentence, extract all instances where a company's name (subjects) and top member employees or key figures (objects) are mentioned together. Exclude general references to employees without specific names or notable roles. Ensure the subject represents a legitimate company or organization, and the object refers to an individual associated with a significant role within the company. Avoid including educational institutions or non-corporate entities as subjects. For each identified relationship: Ensure the subject is a valid company or organization, identifiable by its presence in reliable sources or known business sectors. Confirm the object is a notable individual associated with the company, such as executives, founders. Exclude pronouns and ambiguous references, focusing on clear mentions of both the company and the individual’s name. Format each relationship found in the sentence as follows:  Subject: [COMPANY'S NAME] | Object: [INDIVIDUAL'S NAME]\nSentence: {sentence}"
    }
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
        return response.text
    except Exception as e:
        print(f"\tError during Gemini completion: {str(e)}")
        return ""