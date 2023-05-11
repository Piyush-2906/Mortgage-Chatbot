# -*- coding: utf-8 -*-
"""

"""
from sklearn.metrics.pairwise import cosine_similarity

def chk_similarity(text
                   , vectors
                   , vectorizer
                   , tokenize_facts_df
                   , parent_fact_id
                   , pd
                   , nlp
                   , util
                   , toknm
                   , nltkm
                   , phrase_matcher
                   , txtnm):
    
    child_facts = tokenize_facts_df[(tokenize_facts_df['fact_id'].str.endswith(parent_fact_id))]
    child_facts['cosine_similarity'] = ''
    
    df = pd.DataFrame([text], columns = ['processed_text'])
    df = util.preprocess_html_df(df = df
                                 , nlp = nlp
                                 , nltkm = nltkm
                                 , toknm = toknm
                                 , phrase_matcher = phrase_matcher
                                 , txtnm = txtnm
                                 , is_remove_child = False)
    user_quest = df['processed_text'][0]
    
    test_matrix = vectorizer.transform([user_quest])
    
    for key, val in child_facts.iterrows():
        child_facts['cosine_similarity'][key] = cosine_similarity(test_matrix, vectors[key-1])
 
    child_facts = child_facts.sort_values(["cosine_similarity"], ascending = (False))   
    fact_id = parent_fact_id if child_facts.empty else child_facts['fact_id'].iloc[0] 

    return fact_id