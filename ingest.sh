# Bash script to ingest data
# This involves scraping the data from the web and then cleaning up and putting in Weaviate.
# Error if any command fails
set -e
wget -r -A.html https://python.langchain.com/en/latest/
# wget -r -A.html https://flake8.pycqa.org/en/latest/
python3.10 ingest.py
