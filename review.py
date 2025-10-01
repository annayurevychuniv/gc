#!/usr/bin/env python3
"""
review.py — Vertex AI Code Review bot (Gemini 2.5 Flash)

- Працює через Google Gen AI Python SDK
- З GitHub отримує PR, читає текстові файли
- Робить запит до Gemini та формує коментар
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
GCP_MODEL = os.getenv("GCP_MODEL", "gemini-2.5-flash")  # або gemini-2.5-flash-lite
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
def get_pr_info():
    if GITHUB_REPOSITORY and PR_NUMBER_ENV:
        owner, repo = GITHUB_REPOSITORY.split("/")
        return owner, repo, int(PR_NUMBER_ENV)
    print("ERROR: GITHUB_REPOSITORY or PR_NUMBER not set")
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
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    try:
        r = requests.get(raw_url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        print("Exception fetching raw content:", e)
    return ""

# -------------------
# Helpers: text file detection
# -------------------
TEXT_FILE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".css", ".sh", ".rb", ".php"}

def is_text_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in TEXT_FILE_EXTENSIONS)

# -------------------
# Gemini / GenAI
# -------------------
client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

def call_model(prompt_text: str) -> str:
    try:
        response = client.models.generate_content(
            model=GCP_MODEL,
            content=[{
                "role": "user",
                "content": prompt_text
            }]
        )
        if hasattr(response, "output") and response.output:
            return response.output[0].content
        return "(No content returned)"
    except Exception as e:
        return f"GenAI model call failed: {e}"

# -------------------
# Comment building / posting
# -------------------
def build_comment(reviews: List[dict]) -> str:
    lines = []
    lines.append("## Vertex AI — Automated Code Review (Gemini 2.5 Flash)\n")
    lines.append("I am an automated reviewer. Suggestions below:\n")
    for r in reviews:
        lines.append("---")
        lines.append(f"**File:** `{r['path']}`")
        lines.append("")
        lines.append(r["review"])
        lines.append("")
    lines.append("\n*This is an automated comment.*")
    return "\n".join(lines)

def post_pr_comment(owner: str, repo: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    r = requests.post(url, headers=GH_HEADERS, json={"body": body})
    if r.status_code in (200, 201):
        print("Comment posted to PR")
    else:
        print("Failed to post comment:", r.status_code, r.text)

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
        if len(content) > 20000:
            content = content[:20000] + "\n\n...file truncated..."
        prompt = (
            f"You are a senior software engineer and code reviewer.\n"
            f"Provide concise, actionable review comments as bullet points. "
            f"File: {filename}\n\n```{content}```"
        )
        review_text = call_model(prompt)
        reviews.append({"path": filename, "review": review_text})

    if reviews:
        comment = build_comment(reviews)
        post_pr_comment(owner, repo, pr_number, comment)
    else:
        print("No reviews generated.")

if __name__ == "__main__":
    main()
