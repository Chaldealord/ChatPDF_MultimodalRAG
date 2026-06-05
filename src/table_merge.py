"""
Heuristic merge of adjacent HTML tables that are likely one logical table split across
PDF pages (unstructured emits separate Table elements per page).
"""

from __future__ import annotations

from copy import deepcopy

from lxml import html as lxml_html


def _parse_table_root(html: str) -> lxml_html.HtmlElement | None:
    h = (html or "").strip()
    if not h:
        return None
    try:
        root = lxml_html.fromstring(h)
    except Exception:
        return None
    if root.tag == "table":
        return root
    return root.find(".//table")


def _table_rows(tbl: lxml_html.HtmlElement) -> list[lxml_html.HtmlElement]:
    rows: list[lxml_html.HtmlElement] = []
    for thead in tbl.findall("thead"):
        rows.extend(thead.findall("tr"))
    for tbody in tbl.findall("tbody"):
        rows.extend(tbody.findall("tr"))
    if not rows:
        rows = tbl.findall("tr")
    return rows


def _col_count(tr: lxml_html.HtmlElement) -> int:
    return len(tr.findall("th") + tr.findall("td"))


def _row_cell_texts(tr: lxml_html.HtmlElement) -> tuple[str, ...]:
    out: list[str] = []
    for c in tr.findall("th") + tr.findall("td"):
        txt = (c.text_content() or "").strip()
        out.append(" ".join(txt.split()))
    return tuple(out)


def _target_tbody_for_append(u: lxml_html.HtmlElement) -> lxml_html.HtmlElement:
    tbodies = u.findall("tbody")
    if tbodies:
        return tbodies[-1]

    loose = [c for c in u if c.tag == "tr"]
    tbody = lxml_html.Element("tbody")
    if loose:
        for tr in loose:
            tbody.append(tr)
        u.append(tbody)
        return tbody

    thead = u.find("thead")
    if thead is not None:
        idx = list(u).index(thead) + 1
        u.insert(idx, tbody)
    else:
        u.append(tbody)
    return tbody


def _try_merge_pair(upper_html: str, lower_html: str) -> str | None:
    u = _parse_table_root(upper_html)
    l_tbl = _parse_table_root(lower_html)
    if u is None or l_tbl is None:
        return None

    u_rows = _table_rows(u)
    l_rows = _table_rows(l_tbl)
    if not u_rows or not l_rows:
        return None

    u_head = u_rows[0]
    l_first = l_rows[0]
    if _col_count(u_head) != _col_count(l_first) or _col_count(u_head) < 1:
        return None

    if _row_cell_texts(l_first) == _row_cell_texts(u_head):
        l_trs = l_rows[1:]
    elif l_first.findall("th"):
        return None
    else:
        l_trs = list(l_rows)

    if not l_trs:
        return None

    tbody = _target_tbody_for_append(u)
    for tr in l_trs:
        tbody.append(deepcopy(tr))

    return lxml_html.tostring(u, encoding="unicode", method="html")


def merge_adjacent_split_tables(tables: list[str]) -> list[str]:
    if len(tables) < 2:
        return tables
    out: list[str] = [tables[0]]
    for i in range(1, len(tables)):
        merged = _try_merge_pair(out[-1], tables[i])
        if merged:
            out[-1] = merged
        else:
            out.append(tables[i])
    return out


def merge_adjacent_split_tables_with_pages(
    tables: list[str],
    pages: list[int | None],
) -> tuple[list[str], list[int | None]]:
    """Same merge as `merge_adjacent_split_tables`, but keep one page id per merged table (first segment wins)."""
    if not tables:
        return [], []
    if len(pages) != len(tables):
        pages = [None] * len(tables)
    if len(tables) < 2:
        return tables, list(pages)

    out_t: list[str] = [tables[0]]
    out_p: list[int | None] = [pages[0]]
    for i in range(1, len(tables)):
        merged = _try_merge_pair(out_t[-1], tables[i])
        if merged:
            out_t[-1] = merged
            # Keep primary page of continuing table; prefer first non-null
            if out_p[-1] is None:
                out_p[-1] = pages[i]
        else:
            out_t.append(tables[i])
            out_p.append(pages[i])
    return out_t, out_p
