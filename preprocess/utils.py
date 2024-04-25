import copy


def extract_entity_only(item, mode="eo"):
    new_item = copy.deepcopy(item)

    tokens = item["token"]
    ss, se = item["subj_start"], item["subj_end"]
    _os, oe = item["obj_start"], item["obj_end"]
    subj_span, obj_span = tokens[ss:se + 1], tokens[_os:oe + 1]
    if mode == 'eo':
        pass
    elif mode == 'eo-m':  # mention only
        new_item['subj_type'] = 'SUBJECT_TYPE'
        new_item['obj_type'] = 'OBJECT_TYPE'
    elif mode == 'eo-t':  # type only
        subj_span = ['subject']
        obj_span = ['object']

    new_item["token"] = subj_span + ['and'] + obj_span  # 删除and
    new_item["subj_start"] = 0
    new_item["subj_end"] = len(subj_span) - 1
    new_item["obj_start"] = len(subj_span) + 1
    new_item["obj_end"] = len(subj_span) + len(obj_span)

    return new_item


def extract_context_only(item, mode):
    new_item = copy.deepcopy(item)
    tokens = item['token']
    subj_start, subj_end = item['subj_start'], item['subj_end']
    obj_start, obj_end = item['obj_start'], item['obj_end']

    if subj_start < obj_start:  # subj 出现在 obj之前
        new_tokens = tokens[:subj_start] + ['subject'] + tokens[subj_end + 1:obj_start] + ['object'] + tokens[
                                                                                                       obj_end + 1:]
        new_item['token'] = new_tokens
        new_item['subj_start'] = new_item['subj_end'] = subj_start
        new_item['obj_start'] = new_item['obj_end'] = obj_start - (subj_end - subj_start)
    else:
        new_tokens = tokens[:obj_start] + ['object'] + tokens[obj_end + 1:subj_start] + ['subject'] + tokens[
                                                                                                      subj_end + 1:]
        new_item['token'] = new_tokens
        new_item['obj_start'] = new_item['obj_end'] = obj_start
        new_item['subj_start'] = new_item['subj_end'] = subj_start - (obj_end - obj_start)

    if mode == 'co-o':  # context-only, without entity-type
        new_item['subj_type'] = 'SUBJECT_TYPE'
        new_item['obj_type'] = 'OBJECT_TYPE'

    return new_item


def substitute_item_with_new_entities(item, new_subj, new_obj):
    new_item = copy.deepcopy(item)
    new_subj_span = new_subj.split()
    new_obj_span = new_obj.split()
    ss, se = item["subj_start"], item["subj_end"]
    os, oe = item["obj_start"], item["obj_end"]

    tokens = item["token"]
    new_tokens = []
    new_ss = new_se = 0
    new_os = new_oe = 0

    if ss < os:
        new_tokens.extend(tokens[:ss])
        new_ss = len(new_tokens)
        new_tokens.extend(new_subj_span)
        new_se = len(new_tokens) - 1
        new_tokens.extend(tokens[se + 1:os])
        new_os = len(new_tokens)
        new_tokens.extend(new_obj_span)
        new_oe = len(new_tokens) - 1
        new_tokens.extend(tokens[oe + 1:])
    else:
        new_tokens.extend(tokens[:os])
        new_os = len(new_tokens)
        new_tokens.extend(new_obj_span)
        new_oe = len(new_tokens) - 1
        new_tokens.extend(tokens[oe + 1:ss])
        new_ss = len(new_tokens)
        new_tokens.extend(new_subj_span)
        new_se = len(new_tokens) - 1
        new_tokens.extend(tokens[se + 1:])

    new_item["token"] = new_tokens
    new_item["subj_start"] = new_ss
    new_item["subj_end"] = new_se
    new_item["obj_start"] = new_os
    new_item["obj_end"] = new_oe

    return new_item


def gen_entity_dict(data):
    print("generating entity dict...")
    entity_dict = {}
    for item in data:
        tokens = item["token"]
        ss, se = item["subj_start"], item["subj_end"]
        os, oe = item["obj_start"], item["obj_end"]
        subj_type, obj_type = item["subj_type"], item["obj_type"]
        subj_span, obj_span = tokens[ss:se + 1], tokens[os:oe + 1]
        subj, obj = " ".join(subj_span), " ".join(obj_span)

        if subj_type not in entity_dict:
            entity_dict[subj_type] = set()
        if obj_type not in entity_dict:
            entity_dict[obj_type] = set()

        entity_dict[subj_type].add(subj)
        entity_dict[obj_type].add(obj)

    for key in entity_dict.keys():
        entity_dict[key] = list(entity_dict[key])

    return entity_dict
