from datasets import load_dataset

configs = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages',
           'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
           'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation',
           'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table',
           'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks',
           'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects',
           'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies',
           'word_sorting']
ret = []
for c in configs:
    dataset = load_dataset("lukaemon/bbh", name=c, split="test")
    ret.append((c, set(dataset["target"])))

ret = sorted(ret, key=lambda x: len(x[1]))
for i in ret:
    print(i[0], len(i[1]), i[1])
