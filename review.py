name: Vertex AI Code Review

on:
  push:                           # 🚀 запускатиметься при пуші
    branches:
      - main
      - dev
  pull_request:                   # 🚀 і при PR
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  issues: write
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Configure GCP Service Account
        env:
          GCP_KEY_JSON: ${{ secrets.GCP_KEY_JSON }}
          GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
          GCP_LOCATION: ${{ secrets.GCP_LOCATION }}
          GCP_MODEL: ${{ secrets.GCP_MODEL }}
        run: |
          echo "$GCP_KEY_JSON" > /tmp/gcp-key.json
          echo "GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json" >> $GITHUB_ENV
          echo "GCP_PROJECT_ID=$GCP_PROJECT_ID" >> $GITHUB_ENV
          echo "GCP_LOCATION=$GCP_LOCATION" >> $GITHUB_ENV
          echo "GCP_MODEL=$GCP_MODEL" >> $GITHUB_ENV

      - name: Run Code Review Bot
        env:
          GITHUB_TOKEN: ${{ github.token }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
        run: |
          python review.py
