import os

from flask import Flask, render_template, request
from utils.semantic_search import find_row_numbers, get_df, get_embeddings, top_hits

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", categories=categories)


@app.get("/top")
def top():
    query_ids = request.args.get("id").split()
    if not query_ids:
        return render_template(
            "failure.html", failure_message="Must provide at least one arxivId!"
        )

    set_diff = set(query_ids) - corpus_ids_set
    if set_diff:
        return render_template(
            "failure.html",
            failure_message=f"""
            Could not find the following id(s) in our database: {set_diff}.
            Try again without them!
            """,
        )

    rows = find_row_numbers(query_ids, corpus_ids)
    all_tophits = top_hits(embeddings, rows, int(request.args.get("n_recommendations")))

    hits_formatted = [
        [
            [
                round(paper_index["score"], 3),
                corpus_ids[paper_index["corpus_id"]],
                arxiv_df["title"].values[paper_index["corpus_id"]],
                arxiv_df["abstract"].values[paper_index["corpus_id"]],
            ]
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


path_to_dataset = "../static/arxiv-clean.json"
arxiv_df = get_df(path_to_dataset)
corpus_ids = arxiv_df["id"].values
corpus_ids_set = set(corpus_ids)
categories = sorted(
    list(
        set(
            [
                category
                for categories in arxiv_df["categories"].str.split().values
                for category in categories
            ]
        )
    )
)

filename, _ = os.path.splitext(path_to_dataset)
embeddings = get_embeddings(filename)
