#!/usr/bin/env python3
"""
Vertex AI Code Review Bot using Gemini 2.5 Flash
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
GCP_MODEL = os.getenv("GCP_MODEL", "gemini-2.5-flash")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")
PR_NUMBER = os.getenv("PR_NUMBER")

# -------------------
# GitHub helpers
# -------------------
def get_pr_info():
    if PR_NUMBER and GITHUB_REPOSITORY:
        owner, repo = GITHUB_REPOSITORY.split("/")
        return owner, repo, int(PR_NUMBER)
    print("No PR_NUMBER set — running in debug mode")
    return None, None, None

def list_pr_files(owner, repo, pr_number):
    if not owner:
        return [{"filename":"test.py","raw_url":"https://raw.githubusercontent.com/annayurevychuniv/gc/main/test.py"}]
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        print("Failed to fetch PR files:", r.status_code, r.text)
        return []
    return r.json()

def fetch_raw_content(raw_url):
    r = requests.get(raw_url)
    if r.status_code == 200:
        return r.text
    return ""

# -------------------
# GenAI helper
# -------------------
def genai_review(file_path, file_content):
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    prompt = (
        f"You are a senior software engineer reviewing code. "
        f"Provide concise, actionable review comments for `{file_path}`. "
        f"Highlight bugs, security issues, and style improvements.\n\n"
        f"```{file_content}```"
    )
    try:
        resp = client.models.generate_content(model=GCP_MODEL, contents=prompt)
        # Gemini SDK response text
        return getattr(resp, "text", getattr(resp, "output_text", str(resp)))
    except Exception as e:
        return f"GenAI model call failed: {e}"

# -------------------
# Post comment to PR
# -------------------
def post_pr_comment(body):
    if not PR_NUMBER or not GITHUB_REPOSITORY:
        print("PR_NUMBER not set — skipping GitHub comment")
        print(body)
        return
    owner, repo, pr_number = GITHUB_REPOSITORY.split("/")[0], GITHUB_REPOSITORY.split("/")[1], int(PR_NUMBER)
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.post(url, headers=headers, json={"body": body})
    if r.status_code in (200, 201):
        print("Comment posted to PR")
    else:
        print("Failed to post comment:", r.status_code, r.text)

# -------------------
# Main
# -------------------
def main():
    owner, repo, pr_number = get_pr_info()
    files = list_pr_files(owner, repo, pr_number)
    reviews = []
    for f in files:
        filename = f.get("filename")
        raw_url = f.get("raw_url")
        if not filename or not raw_url:
            continue
        content = fetch_raw_content(raw_url)
        if not content:
            continue
        if len(content) > 25000:
            content = content[:25000] + "\n\n...truncated..."
        review_text = genai_review(filename, content)
        reviews.append(f"**File:** `{filename}`\n{review_text}\n")

    if reviews:
        comment_body = "## Vertex AI — Automated Code Review\n" + "\n---\n".join(reviews)
        post_pr_comment(comment_body)
    else:
        print("No reviews generated.")

if __name__ == "__main__":
    main()
