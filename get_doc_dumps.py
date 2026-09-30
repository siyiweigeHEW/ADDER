#!/usr/bin/env python3
"""Build the documentation dumps that Step 4 reads.

A dump is the operator reference pages of one framework concatenated: each block is
introduced by its URL and separated from the next by a line of 50 `=` characters, which
is what `doc_retriever.py` parses.

The reference index page of each framework is not shipped — put the one you want into
`INDEX` below, and read the root README section 2.2(b) first: the dump has to come from
the version of the specification the frontend under audit implements, or the audit judges
a converter against a specification it was never written for.

Usage:

    python get_doc_dumps.py --out tvm/docxes onnx torch
    python get_doc_dumps.py --out openvino/docxes onnx torch paddle jax

Needs `requests` and `beautifulsoup4`; see `requirements-doc-dumps.txt`.
"""
import argparse
import os
import sys
import time

import requests
from bs4 import BeautifulSoup

SEPARATOR = "=" * 50

# The reference index page of each framework. Any edition works — the vendor's own, a
# localized mirror, a version-pinned snapshot — which is why these are yours to fill in.
INDEX = {
    "onnx": "[index page of the ONNX operator reference]",
    "torch": "[index page of the PyTorch torch.* reference]",
    "paddle": "[index page of the Paddle API reference]",
    "jax": "[index page of the JAX jax.* reference]",
}

# Per framework: which links on the index are operator pages, and where the text sits
# inside one. Selectors are tried in order; the last is the fallback.
FRAMEWORKS = {
    "onnx": (
        lambda href: "onnx__" in href,
        ("section", "body"),
    ),
    "torch": (
        lambda href: "generated/torch." in href,
        ("article", "main", "body"),
    ),
    "paddle": (
        lambda href: "/api/paddle/" in href and "_cn.html" in href and "Overview" not in href,
        ("div.markdown-body", "article", "div[role=main]"),
    ),
    "jax": (
        lambda href: "generated/jax." in href or "_autosummary" in href,
        ("div[role=main]", "article[role=main]", "section"),
    ),
}

HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_TIMEOUT = 20
DELAY_BETWEEN_PAGES = 0.5


def get_soup(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"  [WARN] cannot fetch {url}: {e}")
        return None


def collect_links(index_url, matches):
    soup = get_soup(index_url)
    if soup is None:
        return []
    links = {
        requests.compat.urljoin(index_url, a["href"])
        for a in soup.find_all("a", href=True)
        if matches(a["href"])
    }
    return sorted(links)


def extract_text(soup, selectors):
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            return node.get_text(separator="\n", strip=True)
    return None


def build(name, index_url, out_dir):
    matches, selectors = FRAMEWORKS[name]
    out_path = os.path.join(out_dir, f"{name}doc.txt")

    links = collect_links(index_url, matches)
    print(f"{name}: {len(links)} operator pages")
    if not links:
        print(f"  [WARN] nothing found — is the index page in INDEX the right one?")
        return

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, link in enumerate(links, 1):
            print(f"  [{i}/{len(links)}] {link}")
            page = get_soup(link)
            if page is not None:
                text = extract_text(page, selectors)
                if text:
                    f.write(f"\n{SEPARATOR}\nURL: {link}\n{SEPARATOR}\n")
                    f.write(text)
                    f.write("\n\n")
                    written += 1
            time.sleep(DELAY_BETWEEN_PAGES)

    print(f"{name}: {written}/{len(links)} blocks written to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frameworks", nargs="+", choices=sorted(FRAMEWORKS))
    parser.add_argument("--out", required=True,
                        help="directory to write the dumps into, e.g. tvm/docxes")
    args = parser.parse_args(argv)

    unset = [n for n in args.frameworks if INDEX[n].startswith("[")]
    if unset:
        raise SystemExit(
            "These frameworks still have a placeholder index page in INDEX, so there is "
            "nothing to scrape: " + ", ".join(unset)
        )

    os.makedirs(args.out, exist_ok=True)
    for name in args.frameworks:
        build(name, INDEX[name], args.out)


if __name__ == "__main__":
    main()
