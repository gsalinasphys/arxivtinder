import os

from flask import Flask, render_template, request
from utils.semantic_search import find_row_numbers, get_df, get_embeddings, top_hits

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/top")
def top():
    query_id = request.args.get("id")
    if not query_id:
        return render_template(
            "failure.html",
            failure_message="Must provide at least one arxivId!",
        )

    if query_id not in corpus_ids:
        return render_template(
            "failure.html",
            failure_message=f"Could not find id '{query_id}' in our DB.",
        )

    rows = find_row_numbers([query_id], corpus_ids)
    all_tophits = top_hits(
        embeddings, rows, int(request.args.get("n_recommendations", 10)) + 1
    )[0]

    hits_formatted = [
        [
            round(paper_index["score"], 3),
            corpus_ids[paper_index["corpus_id"]],
            arxiv_df["title"].values[paper_index["corpus_id"]],
            arxiv_df["abstract"].values[paper_index["corpus_id"]],
        ]
        for paper_index in all_tophits
    ]

    return render_template(
        "success.html",
        query_id=query_id,
        hits_formatted=hits_formatted,
    )


path_to_dataset = "../local_files/arxiv-clean.json"
arxiv_df = get_df(path_to_dataset)
corpus_ids = arxiv_df["id"].values
filename, _ = os.path.splitext(path_to_dataset)
embeddings = get_embeddings(filename)
