# -*- coding: utf-8 -*-
"""

"""

def get_matched_rules(rules_df
                      , question_tokens
                      , synonyms_df
                      , nlp
                      , nltkm
                      , pd
                      , rem
                      , txtnm
                      , itertools):
    """
    1.foreach rule
        1.1.foreach token see if exist in question
        1.2.replace token with "1"/"0" according to its exsistance
            1.2.1.if not exsits check for their synonym
        1.3.run the logical operators
        1.4.get the rules which returns 1 along with its no of matched and unmatched count
    """
    try:
        # # add columns to dataframe for logical operation in rule
        # rules_df["logic"] = ""
        # # rules_df["logic"] = rules_df["rule"].copy()
        # # rules_df["logic"] = rules_df["logic"].str.strip()
        # rules_df["logic_output"] = 0
        # rules_df["no_of_matched_tokens"] = 0
        # rules_df["no_of_unmatched_tokens"] = 0

        # # lemmatize/stem synonyms
        # for key, val in synonyms_df.iterrows():
        #     syn_lst = val['synonym'].split(':')
        #     syn_lst = [txtnm.normalize_text(y) for y in syn_lst]
        #     syn = ':'.join(syn_lst)

        #     synonyms_df['synonym'][key] = syn

        # # lemmatize/stem rules
        # for key, val in rules_df.iterrows():
        #     try:
        #         if '<monthly>&<escrow amount>&<determin' in val["rule"]:
        #             print("escres")
        #         rul = val['rule'].strip()
        #         rules_df['logic'][key] = rul
        #         rul = ':'.join([txtnm.normalize_text(y)
        #                        for y in rul.split(':')])
        #         rul = rul.replace("< ", "<").replace(" >", ">")
        #         rules_df['rule'][key] = rul
        #     except Exception as e:
        #         print(
        #             "Exception at get_matched_rules() : >>  rules_df.iterrows() " + str(e))

        for key, value in rules_df.iterrows():
            try:
                # if key == 3:
                #     print("stop")
                lemma_dict = {}
                rule_tokens = value["token"]
                rule_eval = value["logic"]
                
                rule_tokens = rule_tokens.replace(']','').replace('[','').replace("'","").replace('"','').split(",")
                rule_tokens = [x.strip() for x in rule_tokens]
                
                if '(<iowa>|<montana>|<vermont>)>&<cushion>' in value["logic"]:
                    print("escres")
                
                for i in rule_tokens:
                    normalized = txtnm.normalize_text(i)
                    lemma_dict[i] = normalized

                if not ((len(rule_tokens) == 1) & (rule_tokens[0] == "")):
                    for token in rule_tokens:
                        # if 'monthly' == token:
                        #     print("escres")
                        rule_eval = rule_eval.replace("<" + token + ">"
                                                      , isTokenExists(question_tokens
                                                                      , lemma_dict[token]
                                                                      , synonyms_df
                                                                      , pd
                                                                      , rem)
                                                      )
                            
                    rule_logic_output = 1 if eval(rule_eval) else 0  
                    
                    rules_df["logic"][key] = rule_eval
                    rules_df["logic_output"][key] = rule_logic_output        
                    
                    no_of_matched_tokens, no_of_unmatched_tokens = get_no_of_matched_unmatched_tokens(rule_eval)
                    
                    rules_df["no_of_matched_tokens"][key] = no_of_matched_tokens  
                    rules_df["no_of_unmatched_tokens"][key] = no_of_unmatched_tokens
                    
            except Exception as e:
                print("Exception at get_matched_rules() >>  rules_df.iterrows(): " + str(e))      
                print(value["token"])
                print(value["logic"])

    except Exception as e:
        print("Exception at get_matched_rules() : " + str(e))

    return rules_df


def get_no_of_matched_unmatched_tokens(rule):
    return rule.count('1'), rule.count('0')


def isTokenExists(question_tokens, token, synonyms_df, pd, rem):
    try:

        is_token_exists = "0"
        if token in question_tokens:
            is_token_exists = "1"
        else:
            is_token_exists = isTokenSynonymExists(question_tokens, token, synonyms_df, pd, rem)
            
    except Exception as e:
        print("Exception at isTokenExists() : " + str(e))

    return is_token_exists


def isTokenSynonymExists(question_tokens, token, synonyms_df, pd, rem):
    try:

        matched_synonyms = []
        is_token_synonym_exists = "0"

        matched_synonyms_df = pd.DataFrame(columns=["synonym_id", "synonym"])

        # get rows of synonyms
        for key, val in synonyms_df.iterrows():
            synonyms = val['synonym'].split(':')
            if token in synonyms:
                matched_synonyms_df = pd.concat([synonyms_df.loc[[key]]])

        # add synonyms to matched_synonyms
        for key, value in matched_synonyms_df.iterrows():
            synonyms = rem.re.split(r":", value["synonym"])
            matched_synonyms = matched_synonyms + synonyms

        # remove duplicates
        matched_synonyms = list(set(matched_synonyms))

        # get items which are common in "matched_synonyms" and "question_tokens"
        matches = list(set(matched_synonyms) & set(question_tokens))

        if bool(matches):
            is_token_synonym_exists = "1"
        else:
            is_token_synonym_exists = "0"

    except Exception as e:
        print("Exception at isTokenSynonymExists() : " + str(e))

    return is_token_synonym_exists
