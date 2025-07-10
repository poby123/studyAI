import torch
from transformers import BertForNextSentencePrediction
from transformers import AutoTokenizer

model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "pizza is eaten with the use of a knife and fork. In casual settings, however, it is cut into wedges to be eaten while held in the hand."

encoding = tokenizer(prompt, next_sentence, return_tensors='pt')


print('## 1. 임베딩 확인하기')
print('Encoding: ', encoding['input_ids'][0], sep='\n', end='\n\n')
print('Decode for special tokens: ', tokenizer.decode(
    encoding['input_ids'][0]),
    sep="\n", end='\n\n'
)
print('Segment Embedding: ', encoding['token_type_ids'], sep='\n', end='\n\n')


print('## 2. 이어지는 문장을 넣고, 이어지는 문장인지 예측')
pred = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])
probs = torch.nn.functional.softmax(pred.logits, dim=1)
print('Softmax: ', probs)
print('결과: ', ['이어짐', '안이어짐'][torch.argmax(probs, dim=1).item()])

print('\n\n## 3. 안 이어지는 문장을 넣고, 이어지는 문장인지 예측')
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

pred = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])
probs = torch.nn.functional.softmax(pred.logits, dim=1)
print('Softmax: ', probs)
print('결과: ', ['이어짐', '안이어짐'][torch.argmax(probs, dim=1).item()])
