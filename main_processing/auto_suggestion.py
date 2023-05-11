# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from common import word2vec_module as w2m
from common import tfidf_module as tfidfm
from common import fasttext_module as ftm
from common import bert_module as bm

def get_fact_quest(user_ques
                   , questions_df # qna dataframe
                   , toknm
                   , nlp
                   , phrase_matcher
                   , spacy
                   , nltkm
                   , util
                   , tfsm
                   , pd
                   , questions_vectorizer
                   , tfidf_questions_vector
                   , qna_with_synonyms_df # qna with synonyms dataframe
                   , txtnm
                   , config
                   , reloaded_w2v_model
                   , qna_with_synonyms_w2v_features
                   , rem
                   , bert_qna_with_synonyms
                   , model):
    
    try:
        text = user_ques.lower()
        util.remove_punctuations(text)
        similar_questions = []
        questions_df = questions_df[questions_df['ques'].notnull()]
        
        question_tokens = toknm.tokenize(text = text
                                 , nlp = nlp
                                 , phrase_matcher = phrase_matcher
                                 , nltkm = nltkm
                                 , txtnm = txtnm)
        question_tokens_str = [str(x) for x in question_tokens]
        # remove stopwords
        final_ques_tokens = [word for word in question_tokens_str if not word in nltkm.nltk_stopwords]
        
        # if only stopwords are present in user question
        if len(final_ques_tokens) == 0:  
            # string matching from questions_df
            
            questions_df['ques_lower'] = questions_df['ques'].str.lower()
            final_df = questions_df.loc[(questions_df['ques_lower'].str.contains(user_ques.lower()))]
            final_df = final_df.sort_values(by = ['ques_lower'], ascending = True).head(5)
            
            for key, val in final_df.iterrows():
                res = {}
                res['fact_id'] = val['fact_id']
                res['ques'] = val['ques']
                res['tokens'] = []
                
                similar_questions.append(res)
        # if words other than stopwords are present in user question
        else:
            if config.nlp_model == "tfidf":
                
                final_df = tfidfm.get_df_of_similar_questions(questions_vectorizer = questions_vectorizer
                                                              , tfidf_questions_vector = tfidf_questions_vector
                                                              , final_ques_tokens = final_ques_tokens
                                                              , qna_with_synonyms_df = qna_with_synonyms_df
                                                              , util = util
                                                              , tfsm = tfsm
                                                              , pd = pd)
                                              
            elif config.nlp_model == "word2vec":
                final_df = w2m.get_df_of_similar_questions(qna_with_synonyms_w2v_features = qna_with_synonyms_w2v_features
                                                           , user_ques = user_ques
                                                           , nlp = nlp
                                                           , rem = rem
                                                           , txtnm = txtnm
                                                           , np = np
                                                           , reloaded_w2v_model = reloaded_w2v_model
                                                           , util = util
                                                           , tfsm = tfsm
                                                           , pd = pd)
                
            elif config.nlp_model == "fasttext":
                final_df = ftm.get_df_of_similar_questions(user_ques, tfsm)
                
            elif config.nlp_model == "bert":
                final_df = bm.get_df_of_similar_questions(user_ques, bert_qna_with_synonyms, model)
                
                
                
            for key, val in final_df.iterrows():
                res = {}
                res['fact_id'] = val['fact_id']
                res['ques'] = questions_df[(questions_df['fact_id'] == val['fact_id'])
                                 & (questions_df['is_primary_question'] == 1)]['ques'].iloc[0]
                res['tokens'] = []
                similar_questions.append(res)                
    
        return similar_questions
        
    except Exception as e:
        print("Exception in get_fact_quest( ): " + str(e))  
        
