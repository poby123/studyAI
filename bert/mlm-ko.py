# https://wikidocs.net/217116

from transformers import FillMaskPipeline
from transformers import BertForMaskedLM
from transformers import AutoTokenizer

model = BertForMaskedLM.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

inputs = tokenizer('축구는 정말 재미있는 [MASK]다.')
print(inputs)


def print_pip(result: list[dict[str, any]]):
    for item in result:
        print(
            f'{round(item['score'], 2)}, {item['token_str']}, {item['sequence']}'
        )


pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

print()
print_pip(pip('축구는 정말 재미있는 [MASK]다.'))
print()
print_pip(pip('어벤져스는 정말 재미있는 [MASK]다.'))
print()
print_pip(pip('나는 오늘 아침에 [MASK]에 출근을 했다.'))
