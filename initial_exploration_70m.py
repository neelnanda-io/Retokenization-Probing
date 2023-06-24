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
model = HookedTransformer.from_pretrained("pythia-160m")

# %%
data = datasets.load_from_disk("/workspace/Retokenization-Probing/compound_word_dataset")
# %%
data_df = pd.DataFrame({
    "compound_word": data["feature_name"],
    "type": data["label"],
    "index": np.arange(len(data)),
})
# %%
data_df.describe()
# %%
data_df.groupby("compound_word")["type"].value_counts()
# %%
word = "blood-pressure"
sub_df = data_df[data_df.compound_word==word]
A = model.to_single_token(" blood")
B = model.to_single_token(" pressure")
# %%
data_AX = data[sub_df[sub_df.type=="missing_second"].index]["tokens"]
data_AX = data_AX[data_AX[:, -2] == A]
print(len(data_AX)/3200)
data_XB = data[sub_df[sub_df.type=="missing_first"].index]["tokens"]
data_XB = data_XB[data_XB[:, -1] == B]
print(len(data_XB)/3200)
data_AB = data[sub_df[sub_df.type=="bigram"].index]["tokens"]
data_AB = data_AB[(data_AB[:, -1] == B) & (data_AB[:, -2] == A)]
print(len(data_AB)/1600)

# %%
# Run the model over each dataset + save neuron acts
# Save correct log probs
from collections import namedtuple
DataStorage = namedtuple("DataStorage", ["AB", "XB", "AX"])
neuron_acts = DataStorage(
    AB = torch.zeros((len(data_AB), model.cfg.n_layers, model.cfg.d_mlp), dtype=torch.float32, device="cpu"),
    XB = torch.zeros((len(data_XB), model.cfg.n_layers, model.cfg.d_mlp), dtype=torch.float32, device="cpu"),
    AX = torch.zeros((len(data_AX), model.cfg.n_layers, model.cfg.d_mlp), dtype=torch.float32, device="cpu"),
)

A_lps = DataStorage(
    AB = torch.zeros((len(data_AB)), dtype=torch.float32, device="cpu"),
    XB = torch.zeros((len(data_XB)), dtype=torch.float32, device="cpu"),
    AX = torch.zeros((len(data_AX)), dtype=torch.float32, device="cpu"),
)

B_lps = DataStorage(
    AB = torch.zeros((len(data_AB)), dtype=torch.float32, device="cpu"),
    XB = torch.zeros((len(data_XB)), dtype=torch.float32, device="cpu"),
    AX = torch.zeros((len(data_AX)), dtype=torch.float32, device="cpu"),
)

batch_size = 128

for label, dataset in [("AB", data_AB), ("XB", data_XB), ("AX", data_AX)]:
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        # batch = model.to_single_token(batch)
        # with torch.no_grad():
        logits, cache = model.run_with_cache(batch, names_filter=lambda x: x.endswith("hook_pre"), device="cpu")
        neuron_acts.__getattribute__(label)[i:i+batch_size] = einops.rearrange(cache.stack_activation("pre")[:, :, -2, :], "layer batch mlp -> batch layer mlp")
        log_probs = logits.log_softmax(dim=-1)
        A_lps.__getattribute__(label)[i:i+batch_size] = log_probs[:, -3, A]
        B_lps.__getattribute__(label)[i:i+batch_size] = log_probs[:, -2, B]

# %%
lp_df = pd.DataFrame({
    "type": ["AB"]*len(data_AB) + ["XB"]*len(data_XB),
    "A_lp": np.concatenate([to_numpy(A_lps[0]), to_numpy(A_lps[1])]),
    "B_lp": np.concatenate([to_numpy(B_lps[0]), to_numpy(B_lps[1])]),
    "prompt": model.to_string(data_AB)+model.to_string(data_XB)
})
px.histogram(lp_df, x="A_lp", color="type", nbins=100, marginal="rug", hover_name="prompt", barmode="overlay", histnorm="percent").show()
px.histogram(lp_df, x="B_lp", color="type", nbins=100, marginal="rug", hover_name="prompt", barmode="overlay", histnorm="percent").show()

# %%
# Take max mean difference
neuron_df = nutils.make_neuron_df(model.cfg.n_layers, model.cfg.d_mlp)
display(neuron_df.head(5))
# neuron_acts.AB.mean(0).flatten()
# %%
neuron_df["AB_pre"] = to_numpy(neuron_acts.AB.mean(0).flatten())
neuron_df["XB_pre"] = to_numpy(neuron_acts.XB.mean(0).flatten())
neuron_df["AX_pre"] = to_numpy(neuron_acts.AX.mean(0).flatten())
neuron_df["diff_pre"] = neuron_df["AB_pre"] - neuron_df["XB_pre"]
neuron_df["abs_diff_pre"] = neuron_df["diff_pre"].abs()
px.histogram(neuron_df, x=["AB_pre", "XB_pre", "diff_pre"], title="Mean neuron pre-activations", nbins=500, marginal="rug", barmode="overlay", histnorm="percent", hover_name="label", log_y=True, range_x=(-7, 7)).show()

neuron_df["AB_post"] = to_numpy(F.gelu(neuron_acts.AB).mean(0).flatten())
neuron_df["XB_post"] = to_numpy(F.gelu(neuron_acts.XB).mean(0).flatten())
neuron_df["AX_post"] = to_numpy(F.gelu(neuron_acts.AX).mean(0).flatten())
neuron_df["diff_post"] = neuron_df["AB_post"] - neuron_df["XB_post"]
neuron_df["abs_diff_post"] = neuron_df["diff_post"].abs()
px.histogram(neuron_df, x=["AB_post", "XB_post", "diff_post"], title="Mean neuron post-activations", nbins=500, marginal="rug", barmode="overlay", histnorm="percent", hover_name="label", log_y=True, range_x=(-7, 7)).show()
# %%
px.scatter(neuron_df, x="AB_post", y="XB_post", hover_name="label", title="AB vs XB post act").show()
px.scatter(neuron_df, x="AB_pre", y="XB_pre", hover_name="label", title="AB vs XB pre act").show()
# %%
scatter(x=F.gelu(neuron_acts.AB)[::2].mean(0).flatten(), y=F.gelu(neuron_acts.AB)[1::2].mean(0).flatten())
# %%
sorted_neuron_df = neuron_df[neuron_df.L>0].sort_values("diff_post", ascending=False)
sorted_neuron_df
# %%
def plot_dataset_fn(fn):
    temp_df = pd.DataFrame({
        "type": ["AB"]*len(data_AB) + ["XB"]*len(data_XB) + ["AX"]*len(data_AX),
        "value": to_numpy(torch.cat([fn(neuron_acts.AB), fn(neuron_acts.XB), fn(neuron_acts.AX)])),
        "prompt": model.to_string(data_AB)+model.to_string(data_XB)+model.to_string(data_AX)
    })
    return temp_df
# %%
# Plot the top 10 neurons
top_neurons = sorted_neuron_df.head(10)
for i in range(6):
    temp_df = plot_dataset_fn(lambda x: x[:, top_neurons.L.values[i], top_neurons.N.values[i]])
    px.histogram(temp_df, x="value", color="type", nbins=100, marginal="rug", hover_name="prompt", barmode="overlay", histnorm="percent", title=f"Neuron {top_neurons.label.iloc[i]} acts").show()
# %%
px.box(neuron_df, x="L", y="diff_post", hover_name="label", title="Neuron post-act diff").show()
# %%
for i in range(3):
    temp_df = plot_dataset_fn(lambda x: x[:, top_neurons.L.values[i], top_neurons.N.values[i]])
    temp_df["A log prob"] = to_numpy(torch.cat([A_lps[0], A_lps[1], A_lps[2]]))
    px.scatter(temp_df, x="value", y="A log prob", color="type",  hover_name="prompt", title=f"Neuron {top_neurons.label.iloc[i]} acts vs A LP", opacity=0.4, marginal_x="histogram", marginal_y="histogram").show()
    # px.histogram(temp_df, x="value", color="type", nbins=100, marginal="rug", hover_name="prompt", barmode="overlay", histnorm="percent", title=f"Neuron {top_neurons.label.iloc[i]} acts").show()
# %%
for i in range(10):
    temp_df = plot_dataset_fn(lambda x: x[:, top_neurons.L.values[i], top_neurons.N.values[i]])
    temp_df["A log prob"] = to_numpy(torch.cat([A_lps[0], A_lps[1], A_lps[2]]))
    px.scatter(temp_df[(temp_df.type=="AB") & (temp_df["A log prob"]>-5)], x="value", y="A log prob", hover_name="prompt", title=f"Neuron {top_neurons.label.iloc[i]} acts vs A LP", opacity=0.4, marginal_x="histogram", marginal_y="histogram", trendline="ols").show()
# %%
for i in range(5):
    vocab_df = nutils.create_vocab_df(model.W_out[top_neurons.L.iloc[i], top_neurons.N.iloc[i]] @ model.W_U, model=model)
    fig = px.histogram(vocab_df, x="logit", title=top_neurons.label.iloc[i])
    fig.add_vline(vocab_df.loc[B].logit, line_width=3, line_dash="dash", line_color="green")
    fig.add_vline(vocab_df.loc[A].logit, line_width=3, line_dash="dash", line_color="red")
    fig.show()
    temp_df = plot_dataset_fn(lambda x: x[:, top_neurons.L.values[i], top_neurons.N.values[i]])
    px.histogram(temp_df, x="value", color="type", nbins=100, marginal="rug", hover_name="prompt", barmode="overlay", histnorm="percent", title=f"Neuron {top_neurons.label.iloc[i]} acts").show()
# %%
