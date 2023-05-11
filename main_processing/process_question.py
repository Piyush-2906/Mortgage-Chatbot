# -*- coding: utf-8 -*-
"""

"""
# import rule_matcher as rm

def get_fact_obj_by_question(text
                             , meta
                             , toknm
                             , nltkm
                             , spcym
                             , pd
                             , util
                             , tfsm
                             , rem
                             , rm
                             , txtnm
                             , itertools):
    try:
        text = text.lower()
        nlp = spcym.nlp
        phrase_matcher = spcym.phrase_matcher
        
        child_fact_id = ''
        response = []
        child_facts = []
        
        synonyms_df = meta["synonym"]
        rules_df = meta["rules_df"]
        all_rules_df = meta["all_rules_df"]
        phrase_matcher = meta["phrase_matcher"]
        vectors = meta['vectors_model']
        vectorizer = meta['vectorizer']
        tokenize_facts_df = meta["tokenize_facts_df"]
        questions_df = meta['questions_df']
        questions_df = questions_df[questions_df['is_primary_question'] == 1]
        
        question_tokens = toknm.tokenize(text = text
                                         , nlp = nlp
                                         , phrase_matcher = phrase_matcher
                                         , nltkm = nltkm
                                         , txtnm = txtnm)
        question_tokens = [str(x) for x in question_tokens]
        # ----------------------------------------------------------------------------------
        """START : RULE MATCHING """
        # get number of matched and unmatched tokens in question from rules
        rules_df = rm.get_matched_rules(rules_df = rules_df
                                        , question_tokens = question_tokens
                                        , synonyms_df = synonyms_df
                                        , nlp = nlp
                                        , nltkm = nltkm
                                        , pd = pd
                                        , rem = rem
                                        , txtnm = txtnm
                                        , itertools = itertools)
        
        # sort rules_df
        rules_df = rules_df[(rules_df['logic_output'] == 1) | ((rules_df['no_of_unmatched_tokens'] != 0) 
                            & (rules_df['no_of_matched_tokens'] != 0))].sort_values(["logic_output"
                                                                                     , "no_of_unmatched_tokens"
                                                                                     , "no_of_matched_tokens"]
                                                                                     , ascending = (False, True, False))
                                                                                     
        # remove duplicates                                                                         
        rules_df = rules_df.drop_duplicates(subset =['fact_id'
                                                     , 'highlight_div']
                                            , keep = 'first'
                                            , inplace = False)
        
        # get most 5 relevant rules
        matched_rules_df = rules_df.head(5)
        matched_rules_df['child_fact_id'] = ''
        
        """END : RULE MATCHING """
        # ----------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------
        """START : GET CHILD FACT """
        
        # get most similar child div for parent div
        for key, value in  matched_rules_df.iterrows():
            fact_id = value['fact_id']
                    
            if (util.is_child_fact_present(fact_id = fact_id
                                           , tokenize_facts_df = meta["tokenize_facts_df"])):
                
                # Check similarity using tfidf
                child_fact_id = tfsm.chk_similarity(text = text
                                  , vectors = vectors
                                  , vectorizer = vectorizer
                                  , tokenize_facts_df = tokenize_facts_df
                                  , parent_fact_id = fact_id
                                  , pd = pd
                                  , nlp = nlp
                                  , util = util
                                  , toknm = toknm
                                  , nltkm = nltkm
                                  , phrase_matcher = phrase_matcher
                                  , txtnm = txtnm)
                
                matched_rules_df['child_fact_id'][key] = child_fact_id
            else:
                matched_rules_df['child_fact_id'][key] = fact_id
          
        """END : GET CHILD FACT """  
        # ----------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------
        """START : CREATE OUTPUT RESPONSE """ 
              
              
        # create response array
        for key, val in matched_rules_df.iterrows():
    
            # if child_fact_id is already present for other facts(parent fact identified by rules)
            # then do not use similar child_fact_id
            # instead copy parent fact_id to child_fact_id
    
            child_fact_id = val["child_fact_id"]
            
            if child_fact_id in child_facts:
                child_fact_id = val["fact_id"]    
            else:
                child_facts.append(child_fact_id)        
    
            fact_ques_df = questions_df[questions_df['fact_id'] == child_fact_id]
            ques = ""
            ans = ""
            
            fact_arr = fact_ques_df.iloc[0]['fact_id'].split('-')
            
            # if question is not present for child fact then show parent fact question
            for i in reversed(range(len(fact_arr))):
                child_fact = "-".join(fact_arr) # get child_id 
                
                # get ques and ans
                fact_ques_df = questions_df[((questions_df['fact_id'] == child_fact)
                                             &(questions_df['is_primary_question'] == 1))]
                ques = fact_ques_df.iloc[0]['ques']
                ans = fact_ques_df.iloc[0]['ans']
                
                # if question is not present look for parent fact question
                if pd.isna(ques):
                    fact_arr.pop(0) # remove child fact to get parent fact (CF1-)CF1-EBPF1
                else:
                    break
    
            response_obj = {"fact_id" : val["fact_id"], 
                 "document" : val["document"], 
                 "focus_div" : val["focus_div"],
                 "highlight_div" : child_fact_id,
                   "html" : all_rules_df[all_rules_df['fact_id'] == val["focus_div"]].iloc[0]['html'],
                 "header" : val["header"], 
                 "question" : "" if pd.isna(ques) else ques,
                 "answer" : "" if pd.isna(ans) else ans}
            
            # q-fix: for "how does escrow shortage occur?" -- showing two outputs
            is_ques_present = False
            for i in response:
                # print(i['question'])
                quest = "" if pd.isna(ques) else ques
                if quest == i['question']:
                    is_ques_present = True
                    
            if not is_ques_present:        
                response.append(response_obj)
         
        if response == []:
            response.append({"fact_id" : "No match found", 
                 "document" : "No match found", 
                 "focus_div" : "No match found",
                 "highlight_div" : "No match found",
                  "html" : "No match found",
                 "header" : "No match found",
                 "question" : "No match found",
                 "answer" : "No match found"})
    
        """END : CREATE OUTPUT RESPONSE """ 
        # ----------------------------------------------------------------------------------
        
    except Exception as e:
            print("Exception in get_fact_obj_by_question( ): " + str(e))   
            
    return response
    
        
        
    

    
