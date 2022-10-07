import os

from flask import Flask, render_template, request, url_for

from utils.semantic_search import (find_row_numbers, get_df, get_embeddings,
                                   top_hits)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/top", methods=["POST"])
def top():
    path_to_dataset = "../static/arxiv-clean.json"
    arxiv_df = get_df(path_to_dataset)

    filename, _ = os.path.splitext(path_to_dataset)
    corpus_ids = arxiv_df["id"].values
    embeddings = get_embeddings(filename)

    query_ids = request.form.get("id").split()
    if not query_ids or not all(query_id in corpus_ids for query_id in query_ids):
        return render_template("failure.html")

    rows = find_row_numbers(query_ids, corpus_ids)
    all_tophits = top_hits(embeddings, rows)

    hits_formatted = [
        [
            [round(paper_index["score"], 3), corpus_ids[paper_index["corpus_id"]],
            arxiv_df["title"].values[paper_index["corpus_id"]],
            arxiv_df["abstract"].values[paper_index["corpus_id"]]]
            for paper_index in tophits
        ]
        for tophits in all_tophits
    ]

    return render_template(
        "success.html",
        query_ids=query_ids,
        n_queries=len(query_ids),
        hits_formatted=hits_formatted,
    )
