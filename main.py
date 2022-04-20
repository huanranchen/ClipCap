from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
text = "I love China, and"
encoded_input = tokenizer(text, return_tensors='pt')
print(model)

input = torch.randn((5,2,768))

output = model(inputs_embeds = input,)
print(output.logits.shape)
