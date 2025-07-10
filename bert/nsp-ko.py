import torch
from transformers import BertForNextSentencePrediction
from transformers import AutoTokenizer

model = BertForNextSentencePrediction.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "여행을 가보니 한국의 2002년 월드컵 축구대회의 준비는 완벽했습니다."

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
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "극장가서 로맨스 영화를 보고싶어요"
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

pred = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])
probs = torch.nn.functional.softmax(pred.logits, dim=1)
print('Softmax: ', probs)
print('결과: ', ['이어짐', '안이어짐'][torch.argmax(probs, dim=1).item()])
