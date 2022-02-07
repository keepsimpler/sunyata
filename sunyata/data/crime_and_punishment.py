# %%
from sentencepiece import SentencePieceProcessor

# %%
# import os
# print(os.getcwd())
with open('resources/crime-and-punishment-2554.txt') as f:
    text = f.read()
# %%
# The file read above includes metadata and licensing information.
# For training our language model, we will only use the actual novel text.
start = text.find('CRIME AND PUNISHMENT')  # skip header
start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip header
start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip translator preface
end = text.rfind('End of Project')  # skip extra text at the end
text = text[start:end].strip()
len(text)
# %%
Tokenizer = SentencePieceProcessor()
# %%
Tokenizer.load('resources/cp.320.model')
# %%
Tokenizer.vocab_size()
# %%
IDS = Tokenizer.EncodeAsIds(text)  # all the token ids
# %%
type(IDS)
# %%
