import evaluate

input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
perplexity = evaluate.load("perplexity", module_type="metric")

results = perplexity.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=input_texts)
print(results)
