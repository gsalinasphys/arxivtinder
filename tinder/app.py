import os

from flask import Flask, render_template, request
from utils.semantic_search import (find_row_numbers, get_embeddings, get_ids,
                                   top_hits)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/top", methods=["POST"])
def top():
    path_to_dataset = "/home/gsalinas/GitHub/arxivtinder/static/arxiv-clean.json"

    filename, _ = os.path.splitext(path_to_dataset)
    corpus_ids = get_ids(filename)
    embeddings = get_embeddings(filename)

    query_ids = request.form.get("id").split()
    if not query_ids or not all(query_id in corpus_ids for query_id in query_ids):
        return render_template("failure.html")

    rows = find_row_numbers(query_ids, corpus_ids)
    all_tophits = top_hits(embeddings, rows)

    hits_formatted = [
        [
            [round(paper["score"], 3), corpus_ids[paper["corpus_id"]]]
            for paper in tophits
        ]
        for tophits in all_tophits
    ]

    return render_template(
        "success.html",
        query_ids=query_ids,
        n_queries=len(query_ids),
        hits_formatted=hits_formatted,
    )
