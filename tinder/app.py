import os

import numpy as np
from flask import Flask, render_template, request, url_for

from utils.semantic_search import (find_row_numbers, get_df, get_embeddings,
                                   top_hits)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", categories=categories)


@app.route("/top", methods=["POST"])
def top():
    query_ids = request.form.get("id").split()
    if not query_ids or not all(query_id in corpus_ids for query_id in query_ids):
        return render_template("failure.html")

    

    rows = find_row_numbers(query_ids, corpus_ids)
    all_tophits = top_hits(embeddings, rows, int(request.form.get("n_recommendations")))

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
        hits_formatted=hits_formatted
    )

path_to_dataset = "../static/arxiv-clean.json"
arxiv_df = get_df(path_to_dataset)
corpus_ids = arxiv_df["id"].values
categories = sorted(list(set([category for categories in arxiv_df["categories"].str.split().values for category in categories])))

filename, _ = os.path.splitext(path_to_dataset)
embeddings = get_embeddings(filename)
