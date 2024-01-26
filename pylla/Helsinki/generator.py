from transformers import MarianMTModel, MarianTokenizer

def helsinki_generator(prompt):
    model_id = "Helsinki-NLP/opus-mt-fr-en"
    tokenizer = MarianTokenizer.from_pretrained(model_id)

    model = MarianMTModel.from_pretrained(model_id)

    src_text = str(prompt)
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return res[0]
