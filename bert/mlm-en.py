# https://wikidocs.net/217116

from transformers import FillMaskPipeline
from transformers import BertForMaskedLM
from transformers import AutoTokenizer

model = BertForMaskedLM.from_pretrained('bert-large-uncased')
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

inputs = tokenizer('Soccer is a really fun [MASK].')
print('tokenizer result: ', inputs)


def print_pip(result: list[dict[str, any]]):
    for item in result:
        print(
            f'{round(item['score'], 2)}, {item['token_str']}, {item['sequence']}'
        )


pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

print()
print_pip(pip('Soccer is a really fun [MASK].'))
print()
print_pip(pip('The Avengers is a really fun [MASK].'))
print()
print_pip(pip('I went to [MASK] this morning.'))
