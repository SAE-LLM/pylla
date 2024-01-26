from transformers import MarianMTModel, MarianTokenizer


class HelsinkiAI:
    def __init__(self):
        model_id = "Helsinki-NLP/opus-mt-fr-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_id)
        self.model = MarianMTModel.from_pretrained(model_id)
    
    def generate(self, prompt):
        src_text = str(prompt)
        translated = self.model.generate(**self.tokenizer(src_text, return_tensors="pt", padding=True))
        res = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return res[0]
