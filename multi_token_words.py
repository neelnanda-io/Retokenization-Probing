# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

# %%
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_grad_enabled(False)
# %%
# data = load_dataset("/workspace/data/val.jsonl.zst")
model = HookedTransformer.from_pretrained("pythia-70m")
# %%
import requests
import jsonlines
import zstandard
import io

PILE_TEST_URL = 'https://the-eye.eu/public/AI/pile/train/29.jsonl.zst'

# Download the file
response = requests.get(PILE_TEST_URL, stream=True)
response.raise_for_status()  # Ensure we got a valid response

# Prepare a streaming decompression context
dctx = zstandard.ZstdDecompressor()
stream_reader = dctx.stream_reader(io.BytesIO(response.content))

# Wrap the binary stream reader with a TextIOWrapper so jsonlines can read it
text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

lines = []
# Process the JSON lines file
with jsonlines.Reader(text_stream) as reader:
    for obj in reader:
        lines.append({'text': obj['text'], 'subset': obj['meta']['pile_set_name']})
ds = datasets.Dataset.from_list(lines)
# %%
pile_train = utils.tokenize_and_concatenate(ds, model.tokenizer, max_length=512)
pile_train.save_to_disk("/workspace/data/pile_train")
# %%
pile_valid = utils.tokenize_and_concatenate(ds, model.tokenizer, max_length=512)
pile_valid.save_to_disk("/workspace/data/pile_valid")
# pile_valid.save_to_disk("/workspace/data/pile_valid")

pile_both = datasets.concatenate_datasets([pile_valid, pile_data])
# %%
# pile_both = datasets.load_from_disk("/workspace/data/pile_data")
pile_tokens = pile_both["tokens"]
compound_name = "diapers"
A = model.to_single_token(" di")
B = model.to_single_token("apers")
length = 24
dataset_size = 5000
is_AB = (pile_tokens[:, length:-1] == A) & (pile_tokens[:, length+1:] == B)
print((is_AB.sum(dim=-1)).float().sum())
is_XB = (pile_tokens[:, length:-1] != A) & (pile_tokens[:, length+1:] == B)
is_AX = (pile_tokens[:, length:-1] == A) & (pile_tokens[:, length+1:] != B)

# %%
def test_bigram_freq(s):
    print(s)
    print(model.to_str_tokens(s))
    A, B = to_numpy(model.to_tokens(s, prepend_bos=False))[0].tolist()
    is_AB = (pile_tokens[:, :-1] == A) & (pile_tokens[:, 1:] == B)
    print("is_AB", (is_AB.sum()).float())
    is_AX = (pile_tokens[:, :-1] == A) & (pile_tokens[:, 1:] != B)
    print("is_AX", (is_AX.sum()).float())
    is_XB = (pile_tokens[:, :-1] != A) & (pile_tokens[:, 1:] == B)
    print("is_XB", (is_XB.sum()).float())
    print("frac AB | A", (is_AB.sum().float() / (is_AB.sum().float() + is_AX.sum().float())))
    print("frac AB | B", (is_AB.sum().float() / (is_AB.sum().float() + is_XB.sum().float())))

def test_ngram_freq(s):
    print(s)
    print(model.to_str_tokens(s))
    tokens = to_numpy(model.to_tokens(s, prepend_bos=False))[0].tolist()
    n = len(tokens)
    is_X = torch.ones((pile_tokens.shape[0], pile_tokens.shape[1] - (n-1)), dtype=torch.bool)
    for i in range(n):
        prev = is_X.sum()
        if i==n-1:
            is_X = is_X & (pile_tokens[:, i:] == tokens[i])
        else:
            is_X = is_X & (pile_tokens[:, i:-(n-i-1)] == tokens[i])

        print(f"tokens[{i}]")
        print(f"Ratio: {is_X.sum().float() / prev.sum().float():.4%}")
        print(f"Remaining: {is_X.sum()}")
        print()

    # is_AB = (pile_tokens[:, :-1] == A) & (pile_tokens[:, 1:] == B)
    # print("is_AB", (is_AB.sum()).float())
    # is_AX = (pile_tokens[:, :-1] == A) & (pile_tokens[:, 1:] != B)
    # print("is_AX", (is_AX.sum()).float())
    # is_XB = (pile_tokens[:, :-1] != A) & (pile_tokens[:, 1:] == B)
    # print("is_XB", (is_XB.sum()).float())
    # print("frac AB | A", (is_AB.sum().float() / (is_AB.sum().float() + is_AX.sum().float())))
    # print("frac AB | B", (is_AB.sum().float() / (is_AB.sum().float() + is_XB.sum().float())))
test_bigram_freq(" nanogram")
test_ngram_freq(" nanogram")
# %%
def tokens_end_in(A, B):
    is_XB = (pile_tokens[:, :-1] != A) & (pile_tokens[:, 1:] == B)
    value_counts = pd.Series(pile_tokens[:, :-1].flatten()[is_XB.flatten()].tolist()).value_counts()
    for i in range(15):
        print(nutils.process_token(model.to_string(value_counts.index[i])), value_counts.iloc[i])
tokens_end_in(A, B)

# %%
def bool_to_dataset(boolean_array, dataset_size=5000):
    dataset = []
    nonzero_indexes = boolean_array.nonzero()
    batch = nonzero_indexes[:, 0]
    pos = nonzero_indexes[:, 1]
    for b in batch.unique()[:dataset_size]:
        p = boolean_array[b].int().argmax()
        dataset.append(pile_tokens[b, p:p+length+2])
    return torch.stack(dataset)
data_AB = (bool_to_dataset(is_AB))
data_XB = (bool_to_dataset(is_XB))
data_AX = (bool_to_dataset(is_AX))

Path(f"/workspace/data/{compound_name}").mkdir(exist_ok=True)
torch.save(data_AB, f"/workspace/data/{compound_name}/AB.pt")
torch.save(data_XB, f"/workspace/data/{compound_name}/XB.pt")
torch.save(data_AX, f"/workspace/data/{compound_name}/AX.pt")
# %%
