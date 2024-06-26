import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs
from collections import defaultdict


entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
predicates = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}
def spanbert_process(t, r, sentence):
    candidate_pairs = []
    sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
    res = defaultdict(int)
    for ep in sentence_entity_pairs:
        # TODO: keep subject-object pairs of the right type for the target relation (e.g., Person:Organization for the "Work_For" relation)
        if r==1 or r == 2:
          if ep[1][1] == 'PERSON' and ep[2][1] == 'ORGANIZATION':
            candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
          if ep[2][1] == 'PERSON' and ep[1][1] == 'ORGANIZATION':
            candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Subject, e2=Object
        elif r==3:
          if ep[1][1] == 'PERSON' and (ep[2][1] == 'LOCATION' or ep[2][1] == 'CITY' or ep[2][1] == 'STATE_OR_PROVINCE' or ep[2][1] == 'COUNTRY'):
            candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
          if ep[2][1] == 'PERSON' and (ep[1][1] == 'LOCATION' or ep[1][1] == 'CITY' or ep[1][1] == 'STATE_OR_PROVINCE' or ep[1][1] == 'COUNTRY'):
            candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Subject, e2=Object
        elif r==4:
          if ep[1][1] == 'ORGANIZATION' and ep[2][1] == 'PERSON':
            candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
          if ep[2][1] == 'ORGANIZATION' and ep[1][1] == 'PERSON':
            candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Subject, e2=Object
    
    preds = spanbert.predict(candidate_pairs)
    counter = 0
    for ex, pred in list(zip(candidate_pairs, preds)):
            relation = pred[0]
            if relation == 'no_relation':
                continue
            counter += 1
            print("\n\t\t=== Extracted Relation ===")
            print("\t\tTokens: {}".format(ex['tokens']))
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            print("\t\tRelation: {} (Confidence: {:.3f})\nSubject: {}\tObject: {}".format(relation, confidence, subj, obj))
            if confidence > t:
                if res[(subj, relation, obj)] < confidence:
                    res[(subj, relation, obj)] = confidence
                    print("\t\tAdding to set of extracted relations")
                else:
                    print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
            else:
                print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========")
    return res, counter