#!/usr/bin/env python3
"""
review.py — Vertex AI Code Review bot using Google Generative AI SDK

- Отримує PR info з GITHUB_EVENT_PATH або з REPO+PR_NUMBER (локально)
- Лістає файли PR через GitHub API
- Для кожного текстового файлу генерує коментар через Gemini model
- Формує один коментар у PR з усіма оглядами
"""

import os
import sys
import json
import requests
from typing import List

from google import genai

# -------------------
# Config from env
# -------------------
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_MODEL = os.getenv("GCP_MODEL", "gemini-1.5-flash-002")  # оновлено для Gemini
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")
PR_NUMBER_ENV = os.getenv("PR_NUMBER")

# GitHub headers
GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else ""
}

# -------------------
# Helpers: GitHub / PR
# -------------------
def get_event_json():
    ev_path = os.getenv("GITHUB_EVENT_PATH")
    if ev_path and os.path.exists(ev_path):
        with open(ev_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_pr_info():
    ev = get_event_json()
    if ev and "pull_request" in ev:
        owner = ev["repository"]["owner"]["login"]
        repo = ev["repository"]["name"]
        pr_number = ev["pull_request"]["number"]
        return owner, repo, pr_number
    if GITHUB_REPOSITORY and PR_NUMBER_ENV:
        owner, repo = GITHUB_REPOSITORY.split("/")
        return owner, repo, int(PR_NUMBER_ENV)
    print("ERROR: Could not determine PR info.")
    sys.exit(1)

def list_pr_files(owner: str, repo: str, pr_number: int) -> List[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    files = []
    page = 1
    while True:
        r = requests.get(url, headers=GH_HEADERS, params={"per_page": 100, "page": page}, timeout=30)
        if r.status_code != 200:
            print("Failed to fetch PR files:", r.status_code, r.text)
            break
        data = r.json()
        if not data:
            break
        files.extend(data)
        if len(data) < 100:
            break
        page += 1
    return files

def fetch_raw_content(raw_url: str) -> str:
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    try:
        r = requests.get(raw_url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.text
        print(f"Failed to fetch raw content ({raw_url}): {r.status_code}")
    except Exception as e:
        print("Exception fetching raw content:", e)
    return ""

# -------------------
# GenAI Model Call
# -------------------
client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

def call_genai_model(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=GCP_MODEL,
            input=prompt  # <- новий синтаксис SDK
        )
        if hasattr(response, "candidates") and len(response.candidates) > 0:
            return response.candidates[0].content
        return str(response)
    except Exception as e:
        print("GenAI model call failed:", e)
        return f"(GenAI call failed: {e})"

# -------------------
# Comment building / posting
# -------------------
def build_comment(reviews: List[dict]) -> str:
    lines = ["## Vertex AI — Automated Code Review (PoC)\n",
             "I am an automated reviewer. Suggestions below:\n"]
    for r in reviews:
        lines.append("---")
        lines.append(f"**File:** `{r['path']}`\n")
        lines.append(r["review"] + "\n")
    lines.append("*This is an automated comment.*")
    return "\n".join(lines)

def post_pr_comment(owner: str, repo: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    r = requests.post(url, headers=GH_HEADERS, json={"body": body})
    if r.status_code in (200, 201):
        print("Comment posted to PR")
    else:
        print("Failed to post comment:", r.status_code, r.text)

# -------------------
# Utility: detect likely text file
# -------------------
TEXT_FILE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".css", ".sh", ".rb", ".php"}

def is_text_file(filename: str) -> bool:
    low = filename.lower()
    for ext in TEXT_FILE_EXTENSIONS:
        if low.endswith(ext):
            return True
    return True

# -------------------
# Main
# -------------------
def main():
    owner, repo, pr_number = get_pr_info()
    print(f"Reviewing PR: {owner}/{repo}#{pr_number}")

    files = list_pr_files(owner, repo, pr_number)
    if not files:
        print("No files to review")
        return

    reviews = []
    for f in files:
        filename = f.get("filename")
        raw_url = f.get("raw_url")
        if not is_text_file(filename):
            print("Skipping non-text file:", filename)
            continue
        print("Processing", filename)
        content = fetch_raw_content(raw_url)
        if not content:
            print("No content for", filename)
            continue
        MAX_CHARS = 25000
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "\n\n...file truncated for brevity..."
        prompt = (
            "You are a senior software engineer and code reviewer.\n"
            "Provide concise, actionable review comments as bullet points. "
            "Mention potential bugs, security issues, and style improvements. "
            "If you suggest code changes, provide short examples.\n\n"
            f"Review file: {filename}\n\n```{content}```\n"
        )
        review_text = call_genai_model(prompt)
        reviews.append({"path": filename, "review": review_text})

    if not reviews:
        print("No reviews generated.")
        return

    comment = build_comment(reviews)
    post_pr_comment(owner, repo, pr_number, comment)

if __name__ == "__main__":
    main()
