import re
import google.generativeai as genai 


def check_string_regex(s):
    # Pattern to match any of the specified words
    pattern = r'none|not specified|N\/A|Not Available|null|NULL|Not Applicable|NA|Not Provided|Subject: ,|Object: ,|Object: ]|Subject: PERSON\'S NAME|Object: ORGANIZATION|Subject: ORGANIZATION|Object: PERSON\'S NAME|Object: one of LOCATION, CITY, STATE_OR_PROVINCE, or COUNTRY|he|she'
    # Search the string for any match
    return not re.search(pattern, s, re.IGNORECASE)

def get_prompt_text(q, relation_type, sentence):
    # Relation types to specific prompt formats, one can use 1,2,3,4
    # TODO: We may need to get this more refined
    prompts = {
        1: f"Prompt: Given the sentence, extract all instances where person's name (subjects) and educational institutions (objects) are mentioned together. Try to remove pronouns like he/she and ensure person’s name is human name. For each identified relationship: Ensure the subject is a person's name and confirm the object is a legitimate educational institution, identifiable by keywords such University, College, School, or Academy. Format each relationship found in the sentence as follows: Subject: [PERSON'S NAME] | Object: [ORGANIZATION]\nSentence: {sentence}",
        2: f"Given a sentence, extract all names of persons (subjects) and organizations they work for (objects). Extract only when subject and object are there. Give output as this format: [Subject: PERSON'S NAME, Object: ORGANIZATION]\nSentence: {sentence}",
        3: f"Given a sentence, extract all names of persons (subjects) and their living locations (objects). Don't extract pronouns unless it's declared in sentence. Extract only when subject and object are there.  Give output as this format: [Subject: PERSON'S NAME, Object: one of LOCATION, CITY, STATE_OR_PROVINCE, or COUNTRY]\nSentence: {sentence}",
        4: f"Given a sentence, extract all organizations (subjects) and top member employees (objects). Don't extract pronouns unless it's declared in sentence. Extract only when subject and object are there.  Give output as this format: [Subject: ORGANIZATION, Object: PERSON'S NAME]\nSentence: {sentence}"
    }
    return prompts.get(relation_type, "Invalid relation type.")

# Function to get content generation from the Gemini API
def get_gemini_completion(prompt_text, api_key, model_name='gemini-pro', max_tokens=100, temperature=0.5, top_p=1, top_k=32):
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
