# Semantics

Semantic similarity in Elixir using text embeddings from the excellent Python library [SentenceTransformers by SBert](https://www.sbert.net/index.html#).

This is a very simple library that provides an `erlport`-based wrapper to SentenceTransformers, and a cosine similarity helper from [Similarity](https://github.com/preciz/similarity).

## Example

```
iex(1)> import Semantics
Semantics
iex(2)> start_link("paraphrase-MiniLM-L6-v2")   # See SentenceTransformer docs for full list
python start args: [
  env: [{'VIRTUAL_ENV', '/home/lookpop/semantics/priv/python/semantics-venv'}],
  python: '/home/lookpop/semantics/priv/python/semantics-venv/bin/python3',
  python_path: '/home/lookpop/semantics/priv/python'
]
SEMANTICS: loading model paraphrase-MiniLM-L6-v2
{:ok, #PID<0.206.0>}
iex(3)> embedding("I like cats")
[0.23906055092811584, -1.1417245864868164, 0.13355520367622375,
 0.13051727414131165, -0.6010502576828003, 0.20810797810554504,
 0.9089261293411255, -0.001883262419141829, -0.044903531670570374,
 0.2549824118614197, -0.5482040047645569, -0.7193037867546082,
 0.12138155847787857, 0.24462690949440002, 0.3153916895389557,
 0.13613221049308777, 0.7277143597602844, -0.13291320204734802,
 -0.06399975717067719, -0.28735366463661194, -0.7334134578704834,
 -0.35985904932022095, -0.1697186678647995, 0.3418505787849426,
 -0.8475354313850403, -0.1252552568912506, -0.32450196146965027,
 0.2670220136642456, -0.28907573223114014, -0.2645415961742401,
 0.05238057300448418, -0.29865625500679016, 0.05948035791516304,
 -0.7136659026145935, -0.3152972161769867, -0.11816924810409546,
 0.02663307823240757, -0.20642021298408508, -0.45193952322006226,
 -0.15293395519256592, -0.2800045609474182, -0.2381720095872879,
 0.49682706594467163, -0.07594038546085358, 0.24341261386871338,
 -0.5986779928207397, 0.011733309365808964, -0.5240899324417114,
 0.7714636921882629, 0.7268072366714478, ...]
iex(4)> similarity(embedding("I like cats"), embedding("I like kittens"))
0.907135858166963
iex(5)> similarity(embedding("I like cats"), embedding("I like dogs"))
0.6468114092540255
iex(6)> similarity(embedding("I like cats"), embedding("I like fiduciary responsibility"))
0.20907087175155692
iex(7)> similarity(embedding("I want to go horseback riding"), embedding("I want to do equestrian stuff"))
0.7922539232253905
iex(8)> similarity(embedding("I want to go horseback riding"), embedding("Riding crops and saddles"))
0.5468014070228772
iex(9)> similarity(embedding("I want to go horseback riding"), embedding("Spaceship parts and electric cars"))
0.033702825222684064
```

## Usage notes

You can do `start_link()` without an argument to use the default model, `paraphrase-MiniLM-L6-v2`.

Refer to [erlport's extensive documentation](http://erlport.org/docs/) for some of the finer points
of wiring BEAM into Python. 

If you want to see the Python side of things, see `priv/python/app.py`.

See "Warning", below.

## Available models

Quite a few. See [SentenceTransformers pretrained models list](https://www.sbert.net/docs/pretrained_models.html).

## Warning

Important: The first time Semantics starts, it will try to setup a venv for use in its own deps/semantics/priv/python folder.
The requisite Python libraries are almost 1GB. Do not be alarmed by long start times during first initialization.

If you want to take these steps by hand, the Elixir code will skip autoinstallation. Here's how:

```
my_app$ cd deps/semantics
my_app/deps/semantics$ cd priv/python
my_app/deps/semantics/priv/python$ python3 -m venv semantics-venv
my_app/deps/semantics/priv/python$ source semantics-venv/bin/activate
(semantics-venv) my_app/deps/semantics/priv/python$ python3 -m pip install -r requirements.txt
```

## Installation

```elixir
def deps do
  [
    {:semantics, git: "https://github.com/tlack/semantics"}
  ]
end
```

## Evaluating models

There is a Python-level script available to use to evaluate different models against your task.

The evaluator accepts named groups of tests. Each test consists of two pairs of texts - one 
that should evaluate closely together in the embedding space, and the other pair that should be further apart.

The evaluator will try all the models you've configured against your tests and report results.

See `priv/python/evaluate.py` and its corresponding files 

## Fine tuning by retraining models

What do you do if the models don't work?

There is some code in `priv/python/retrain.py` that shows how you can use SentenceTransformer's
retraining system. It requires labeled pairs of texts, where the label is a similarity score.

# Credits and Contact

Need help? Want to discuss NLP in Elixir? lackner@gmail.com
