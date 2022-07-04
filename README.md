# arxivtinder

Recommendation system for arXiv papers

## Dev Setup

- Create a python `3.10.4` virtual environment using your favorite method
- Install the latest `pip-tools`
  `pip install pip-tools --upgrade`
- Install test dependencies
  `pip install -r requirements/test_requirements.txt`
- Install pre-commit hooks
  `pre-commit install`

## Adding/Removing/Upgrading dependencies

We use `pip-compile` to generate the `requirements/*.txt` files. To update them, edit the appropriate `.in` file and then run the following command from the project root

`pip-compile requirements/requirements.in && pip-compile requirements/test_requirements.in`
