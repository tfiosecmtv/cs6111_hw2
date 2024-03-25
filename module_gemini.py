import re
import google.generativeai as genai 


def check_string_regex(s):
    # Pattern to match any of the specified words
    pattern = r'none|not specified|N\/A|Not Available|null|NULL|Not Mentioned|Not Applicable|NA|Not Provided|Subject: ,|Object: ,|Object: ]|PERSON\'S NAME|ORGANIZATION|LOCATION|CITY|STATE_OR_PROVINCE|COUNTRY|he|she|NOT FOUND|Subject: \|'
    # Search the string for any match
    return not re.search(pattern, s, re.IGNORECASE)

def get_prompt_text(q, relation_type, sentence):
    # Relation types to specific prompt formats, one can use 1,2,3,4
    # TODO: We may need to get this more refined
    prompts = {
        1: f"Prompt: Given the sentence, extract all instances where person's name (subjects) and educational institutions (objects) are mentioned together. For objects or subjects, do not include companies like Google/Microsoft as they are not educational institution. Try to remove pronouns like he/she and ensure person’s name is human name. For each identified relationship: Ensure the subject is a person's name and confirm the object is a legitimate educational institution, identifiable by keywords such University, College, School, or Academy. Format each relationship found in the sentence as follows: Subject: [PERSON'S NAME] | Object: [ORGANIZATION]\nSentence: {sentence}",
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
