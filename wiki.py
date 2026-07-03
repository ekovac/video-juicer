"""Read episode plot summaries from a local Wikipedia dump.

Uses the **multistream** dump pair published by Wikimedia:
  enwiki-<date>-pages-articles-multistream.xml.bz2        (data)
  enwiki-<date>-pages-articles-multistream-index.txt.bz2  (index)

The data file is a concatenation of independent ~100-page bz2 streams; the index
maps `offset:pageid:title` per page. A lookup is therefore: find the title's
offset in the index, seek there, decompress exactly ONE stream, and pull the
page out of the ~100 it contains. No network, no extra dependencies — just bz2.

The index stores byte offsets into one specific data file, so the two MUST come
from the same dump run; a mismatch makes the seek land mid-stream and bz2
decompression raises (surfaced as a clear error, not silent garbage).
"""
from __future__ import annotations

import bz2
import html
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# multistream reader
# ---------------------------------------------------------------------------


class SnapshotError(RuntimeError):
    """Reading the dump failed (missing file, or index/data run mismatch)."""


class MultistreamSnapshot:
    def __init__(self, data_path: str | Path, index_path: str | Path):
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        if not self.data_path.is_file():
            raise SnapshotError(f"dump not found: {self.data_path}")
        if not self.index_path.is_file():
            raise SnapshotError(f"index not found: {self.index_path}")

    def _find_offset(self, title: str) -> Optional[int]:
        """Byte offset of the bz2 stream holding `title`. The index is a single
        bz2 stream ordered by page id (not title), so this is a linear scan —
        fine for the handful of article lookups we do per show."""
        want = title.replace("_", " ")
        with bz2.open(self.index_path, "rt", encoding="utf-8") as f:
            for line in f:
                # offset:pageid:title  (title may itself contain ':')
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) == 3 and parts[2] == want:
                    return int(parts[0])
        return None

    def _stream_at(self, offset: int) -> str:
        """Decompress exactly one bz2 stream starting at `offset`."""
        dec = bz2.BZ2Decompressor()
        out: list[bytes] = []
        try:
            with open(self.data_path, "rb") as f:
                f.seek(offset)
                while not dec.eof:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    out.append(dec.decompress(chunk))
        except OSError as e:
            raise SnapshotError(
                f"bz2 decompression failed at offset {offset} — the index and "
                f"data file are probably from different dump runs: {e}") from e
        return b"".join(out).decode("utf-8", "replace")

    def article(self, title: str, _hops: int = 2) -> Optional[str]:
        """Wikitext of `title`, following #REDIRECT (up to `_hops`). None if the
        title isn't in the dump."""
        offset = self._find_offset(title)
        if offset is None:
            return None
        text = _extract_page_text(self._stream_at(offset), title)
        if text is None:
            return None
        m = re.match(r"\s*#REDIRECT\s*\[\[([^\]|#]+)", text, re.IGNORECASE)
        if m and _hops > 0:
            return self.article(m.group(1).strip(), _hops - 1)
        return text


def _extract_page_text(xml: str, title: str) -> Optional[str]:
    """Pull one page's <text> out of a decompressed multistream block (~100
    pages of `<page>…</page>`), matched by <title>."""
    want = title.replace("_", " ")
    for page in re.finditer(r"<page>(.*?)</page>", xml, re.DOTALL):
        body = page.group(1)
        mt = re.search(r"<title>(.*?)</title>", body, re.DOTALL)
        if not mt or html.unescape(mt.group(1)) != want:
            continue
        mx = re.search(r"<text\b[^>]*>(.*?)</text>", body, re.DOTALL)
        return html.unescape(mx.group(1)) if mx else ""
    return None


# ---------------------------------------------------------------------------
# episode-summary parsing ({{Episode list}} templates)
# ---------------------------------------------------------------------------

_SEASON_HEADER = re.compile(r"^=+\s*(?:Season|Series)\s+(\d+)", re.IGNORECASE | re.M)


def _clean(s: str) -> str:
    """Strip wiki markup from a ShortSummary into plain prose."""
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.DOTALL)   # cite refs
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)                          # simple templates
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)       # [[a|b]] -> b
    s = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", s)          # [url label]
    s = re.sub(r"'''?", "", s)                                    # bold/italic
    s = re.sub(r"<[^>]+>", "", s)                                 # stray html
    return re.sub(r"\s+", " ", s).strip()


def _episode_list_blocks(wikitext: str):
    """Yield (season, raw_template) for each {{Episode list …}}, tracking the
    current `== Season N ==` header so within-season numbering can be resolved."""
    season = 1
    i = 0
    while i < len(wikitext):
        hdr = _SEASON_HEADER.search(wikitext, i)
        tmpl = wikitext.find("{{Episode list", i)
        if tmpl < 0:
            return
        if hdr and hdr.start() < tmpl:      # a header comes first — adopt it
            season = int(hdr.group(1))
            i = hdr.end()
            continue
        # balanced-brace scan of the template
        depth, k = 0, tmpl
        while k < len(wikitext):
            if wikitext[k:k + 2] == "{{":
                depth += 1
                k += 2
            elif wikitext[k:k + 2] == "}}":
                depth -= 1
                k += 2
                if depth == 0:
                    break
            else:
                k += 1
        yield season, wikitext[tmpl:k]
        i = k


def _template_params(block: str) -> dict[str, str]:
    """Top-level named params of a template, respecting nested {{}} and [[]]."""
    body = block[2:-2]
    parts, depth, cur, i = [], 0, "", 0
    while i < len(body):
        two = body[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            cur += two
            i += 2
        elif two in ("}}", "]]"):
            depth -= 1
            cur += two
            i += 2
        elif body[i] == "|" and depth == 0:
            parts.append(cur)
            cur = ""
            i += 1
        else:
            cur += body[i]
            i += 1
    parts.append(cur)
    out = {}
    for p in parts[1:]:                     # parts[0] is the template name
        if "=" in p:
            k, _, v = p.partition("=")
            out[k.strip()] = v
    return out


def parse_episode_summaries(wikitext: str) -> dict[tuple[int, int], tuple[str, str]]:
    """{(season, number): (title, summary)} from a 'List of … episodes' article.

    Numbering uses the template's within-season `EpisodeNumber2` when present,
    else a per-season running counter; the season comes from the enclosing
    `== Season N ==` header. Only entries carrying a ShortSummary are returned."""
    out: dict[tuple[int, int], tuple[str, str]] = {}
    counters: dict[int, int] = {}
    for season, block in _episode_list_blocks(wikitext):
        d = _template_params(block)
        summary = _clean(d.get("ShortSummary", ""))
        if not summary:
            continue
        num = None
        raw = (d.get("EpisodeNumber2") or "").strip()
        m = re.match(r"\d+", raw)
        if m:
            num = int(m.group(0))
        if num is None:
            num = counters.get(season, 0) + 1
        counters[season] = num
        out[(season, num)] = (_clean(d.get("Title", "")), summary)
    return out


def episode_summaries(snapshot: MultistreamSnapshot, page_title: str
                      ) -> dict[tuple[int, int], tuple[str, str]]:
    """Fetch a 'List of … episodes' article from the dump and parse it. Raises
    SnapshotError if the page isn't found."""
    wt = snapshot.article(page_title)
    if wt is None:
        raise SnapshotError(f"page not found in dump: {page_title!r}")
    return parse_episode_summaries(wt)
