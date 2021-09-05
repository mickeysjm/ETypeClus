import spacy
from collections.abc import Iterable

nlp = spacy.load('en_core_web_lg')

# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}


# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet, depTagSet):
    lexical_match = ("and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet)
    dependency_match = ("conj" in depTagSet)
    return lexical_match or dependency_match

def contains_conj_strict(depSet, depTagSet):
    lexical_match = ("and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet)
    return lexical_match


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        rightDepTags = {tok.dep_ for tok in rights}
        if contains_conj(rightDeps, rightDepTags):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN" or tok.dep_ == "conj"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        rightDepTags = {tok.dep_ for tok in rights}
        if contains_conj_strict(rightDeps, rightDepTags):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ == "NOUN":
        return [head], _is_negated(tok)
    return [], False


# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False


# get all the verbs on tokens with negation marker
def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# xcomp; open complement - verb has no suject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


# find the main verb - or any aux verb if we can't find it
def _find_verbs(tokens):
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    if len(verbs) == 0:
        verbs = [tok for tok in tokens if _is_verb(tok)]
    return verbs


# is the token an auxiliary verb?
def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass")


# is the token a verb?  (including auxiliary verbs)
def _is_verb(tok):
    return tok.pos_ == "VERB" or tok.pos_ == "AUX"


# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    #potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs


# return true if the sentence is passive - at the moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


# return true if the current verb is passive
def _is_cur_v_passive(v):
    for tok in v.children:
        if tok.dep_ == "auxpass":
            return True
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return None


# simple stemmer using lemmas
def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


# print information for displaying all kinds of things of the parse tree
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


# expand an obj / subj np using its chunk
def expand(item, tokens, visited):
    if item.lower_ == 'that':
        temp_item = _get_that_resolution(tokens)
        if temp_item is not None:
            item = temp_item

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN":
                if item2.i not in visited:
                    visited.add(item2.i)
                    parts.extend(expand(item2, tokens, visited))
            break

    return parts


# convert a list of tokens to a string
def to_str(tokens):
    if isinstance(tokens, Iterable):
        return [' '.join([item.text for item in tokens]), [item.i for item in tokens]]
    else:
        return ['', [-1]]


# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens):
    svos = []
    svos_indexs = []
    verbs = _find_verbs(tokens)
    visited = set()  # recursion detection
    for v in verbs:
        is_pas = _is_cur_v_passive(v)
        subs, verbNegated = _get_all_subs(v)
        # hopefully there are subs
        if len(subs) > 0:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        expanded_sub = expand(sub, tokens, visited)
                        expanded_obj = expand(obj, tokens, visited)
                        normalized_verb1 = "!" + v.lemma_ if verbNegated or objNegated else v.lemma_
                        normalized_verb2 = "!" + v2.lemma_ if verbNegated or objNegated else v2.lemma_
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expanded_obj), [normalized_verb1, v.i], to_str(expanded_sub)))
                            svos.append((to_str(expanded_obj), [normalized_verb2, v2.i], to_str(expanded_sub)))
                        else:
                            svos.append((to_str(expanded_sub), [normalized_verb1, v.i],  to_str(expanded_obj)))
                            svos.append((to_str(expanded_sub), [normalized_verb2, v2.i],  to_str(expanded_obj)))
            else:
                v, objs = _get_all_objs(v, is_pas)
                for sub in subs:
                    if len(objs) > 0:
                        for obj in objs:
                            objNegated = _is_negated(obj)
                            expanded_sub = expand(sub, tokens, visited)
                            expanded_obj = expand(obj, tokens, visited)
                            normalized_verb = "!" + v.lemma_ if verbNegated or objNegated else v.lemma_
                            if is_pas:  # reverse object / subject for passive
                                svos.append((to_str(expanded_obj), [normalized_verb, v.i], to_str(expanded_sub)))
                            else:
                                svos.append((to_str(expanded_sub), [normalized_verb, v.i], to_str(expanded_obj)))
                    else:
                        # no obj - just return the SV parts
                        expanded_sub = expand(sub, tokens, visited)
                        normalized_verb = "!" + v.lemma_ if verbNegated else v.lemma_
                        if is_pas:
                            svos.append((None, [normalized_verb, v.i], to_str(expanded_sub)))
                        else:
                            svos.append((to_str(expanded_sub), [normalized_verb, v.i], None))

        else:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for obj in objs:
                    objNegated = _is_negated(obj)
                    expanded_obj = expand(obj, tokens, visited)
                    normalized_verb1 = "!" + v.lemma_ if verbNegated or objNegated else v.lemma_
                    normalized_verb2 = "!" + v2.lemma_ if verbNegated or objNegated else v2.lemma_
                    
                    if is_pas:  # reverse object / subject for passive
                        svos.append((to_str(expanded_obj), [normalized_verb1, v.i], None))
                        svos.append((to_str(expanded_obj), [normalized_verb2, v2.i], None))
                    else:
                        svos.append((None, [normalized_verb1, v.i], to_str(expanded_obj)))
                        svos.append((None, [normalized_verb2, v2.i], to_str(expanded_obj)))
            else:
                v, objs = _get_all_objs(v, is_pas)
                if len(objs) > 0:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        expanded_obj = expand(obj, tokens, visited)
                        normalized_verb = "!" + v.lemma_ if verbNegated or objNegated else v.lemma_
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expanded_obj), [normalized_verb, v.i], None))
                        else:
                            svos.append((None, [normalized_verb, v.i], to_str(expanded_obj)))

    return svos


if __name__ == '__main__':
    test_cases = [
        [0, 4642, "Scientists are still investigating this branch of the outbreak , trying to establish if the virus - perhaps present in human faeces - was spread by vermin or by leaking sewage along a vertical shaft in the building ."],
        [1, 9406, "NEW DELHI , India ( CNN ) -- The death toll from an outbreak of hepatitis B in India 's western Gujarat state reached 38 on Sunday as authorities prepared to begin a vaccination drive against the disease ."],
        [2, 42, "For assistance , please send e - mail to : mmwrq@cdc.gov ."],
        [3, 2021, "The index patient ( patient A ) had onset of symptoms on February 15 ."],
        [4, 74944, "China , Singapore , Malaysia , Indonesia and Vietnam continue to execute people who use drugs in high numbers ."],
        [5, 133, "These conditions were blamed for the return of plague in India after 28 years — the last epidemic was registered in 1966 ."],
        [6, 46127, "The outbreak began in Kozhikode district and later spread to the adjoining Malappuram district ."],
        [7, 35594, "Last month , President Dilma Rousseff mobilized 220,000 troops to give out leaflets and help householders eliminate breeding grounds of the mosquito that spreads Zika ."],
        [8, 2198, "As of March 25 , a cluster of 13 persons with suspected / probable SARS are known to have stayed at hotel M ( Figure 1 ) ."],
        [9, 5148, "\" When they [ the prosecutors ] decide who should bear the responsibility [ for the outbreak ] , we 'll know who to blame as well ."]
    ]

    for test_case in test_cases:
        raw_sent = test_case[2]
        doc = nlp(raw_sent)
        print(raw_sent)
        print([(tok.text, tok.dep_) for tok in doc])
        svos = findSVOs(doc)
        print("svos:", svos)
        print("="*80)
