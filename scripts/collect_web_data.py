"""
Collect and refresh data from Sanchit's online profiles.
Saves enriched markdown files to data/knowledge_base/.

Run:
    python scripts/collect_web_data.py

Then rebuild the index:
    python scripts/build_index.py
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


GITHUB_USERNAME = "SanchitMinocha"
GITHUB_API = "https://api.github.com"


def fetch_github_repos() -> list:
    logger.info("Fetching GitHub repositories...")
    url = f"{GITHUB_API}/users/{GITHUB_USERNAME}/repos?per_page=50&sort=updated"
    try:
        resp = requests.get(url, timeout=10, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()
        repos = resp.json()
        logger.info(f"Found {len(repos)} repositories")
        return repos
    except Exception as e:
        logger.warning(f"Failed to fetch GitHub repos: {e}")
        return []


def fetch_github_profile() -> dict:
    logger.info("Fetching GitHub profile...")
    try:
        resp = requests.get(f"{GITHUB_API}/users/{GITHUB_USERNAME}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch GitHub profile: {e}")
        return {}


def build_github_markdown(profile: dict, repos: list) -> str:
    lines = [
        "# Sanchit Minocha – GitHub Profile & Projects",
        "",
        f"**GitHub Username:** {GITHUB_USERNAME}",
        f"**Profile URL:** https://github.com/{GITHUB_USERNAME}",
    ]
    if profile:
        lines += [
            f"**Bio:** {profile.get('bio', 'N/A')}",
            f"**Location:** {profile.get('location', 'N/A')}",
            f"**Company:** {profile.get('company', 'N/A')}",
            f"**Public Repos:** {profile.get('public_repos', 0)}",
            f"**Followers:** {profile.get('followers', 0)}",
            f"**Website:** {profile.get('blog', 'N/A')}",
            "",
        ]

    lines += ["## Repositories", ""]

    for repo in repos:
        if repo.get("fork"):
            continue  # skip forked repos
        name = repo.get("name", "")
        desc = repo.get("description") or "No description"
        lang = repo.get("language") or "N/A"
        stars = repo.get("stargazers_count", 0)
        url = repo.get("html_url", "")
        updated = repo.get("updated_at", "")[:10] if repo.get("updated_at") else "N/A"

        lines += [
            f"### {name}",
            f"- **Description:** {desc}",
            f"- **Language:** {lang}",
            f"- **Stars:** {stars}",
            f"- **URL:** {url}",
            f"- **Last Updated:** {updated}",
            "",
        ]

    lines += [
        f"---",
        f"*Last fetched: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
    ]
    return "\n".join(lines)


def save_github_data(profile: dict, repos: list):
    content = build_github_markdown(profile, repos)
    path = config.KNOWLEDGE_BASE_DIR / "github.md"
    path.write_text(content, encoding="utf-8")
    logger.info(f"Saved GitHub data to {path}")


def save_summary():
    """Save a summary file that lists all data sources and last update time."""
    summary = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "sources": {
            "github": f"https://github.com/{GITHUB_USERNAME}",
            "website": "https://sanchitminocha.github.io/",
            "linkedin": "https://www.linkedin.com/in/sanchitminochaiitr/",
        },
        "note": "Run collect_web_data.py to refresh, then build_index.py to update the RAG index."
    }
    path = config.DATA_DIR / "sources_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved sources summary to {path}")


def main():
    logger.info("Starting data collection for SanchitAI...")

    profile = fetch_github_profile()
    repos = fetch_github_repos()

    if repos:
        save_github_data(profile, repos)
    else:
        logger.warning("No GitHub data fetched — skipping github.md update")

    save_summary()
    logger.info("Data collection complete.")
    logger.info("Next step: Run `python scripts/build_index.py` to rebuild the RAG index.")


if __name__ == "__main__":
    main()
