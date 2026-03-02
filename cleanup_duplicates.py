# -*- coding: utf-8 -*-
"""Remove duplicate sections inserted by multiple upgrade_report.py runs."""

from docx import Document
from docx.oxml.ns import qn

doc = Document("wfm \u90e8\u5206.docx")
body = doc.element.body

def get_text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))

elems = list(body)

# Identify duplicate blocks to remove
# Each entry: keyword that identifies the START of a duplicate block, and how many elements to remove
# We need to remove the SECOND occurrence of each block

def find_nth(keyword, n=2):
    """Find nth occurrence (1-indexed) of paragraph with keyword"""
    count = 0
    for i, el in enumerate(list(body)):
        if el.tag.endswith("}p") and keyword in get_text(el):
            count += 1
            if count == n:
                return i, el
    return None, None

# ── Remove duplicate KMO/FL section (second occurrence) ──────────────────────
# First occurrence: [139] "（一）KMO取样适切性检验..."
# Second occurrence: [152] "（一）KMO取样适切性检验..." (DUPLICATE)
# The second occurrence spans from [152] through [164]

# Find start of second KMO block
idx2_kmo, el2_kmo = find_nth("KMO\u53d6\u6837\u9002\u5207\u6027\u68c0\u9a8c\u4e0eBartlett\u7403\u578b\u68c0\u9a8c", n=2)

# Find where the second block ends (should end before "一、样本基本特征")
idx_end_kmo, el_end_kmo = None, None
if idx2_kmo is not None:
    elems2 = list(body)
    for i in range(idx2_kmo, min(idx2_kmo + 30, len(elems2))):
        el = elems2[i]
        t = get_text(el) if el.tag.endswith("}p") else ""
        if "\u4e00\u3001\u6837\u672c\u57fa\u672c\u7279\u5f81" in t:  # "一、样本基本特征"
            idx_end_kmo = i
            break

print(f"Duplicate KMO block: [{idx2_kmo}] -> [{idx_end_kmo})")

# Remove the duplicate block
if idx2_kmo is not None and idx_end_kmo is not None:
    elems3 = list(body)
    to_remove = elems3[idx2_kmo:idx_end_kmo]
    for el in to_remove:
        body.remove(el)
    print(f"  Removed {len(to_remove)} elements (KMO/FL duplicate)")
else:
    print("  KMO duplicate not found or already clean")

# ── Remove duplicate seat analysis (second occurrence) ─────────────────────
idx2_seat, el2_seat = find_nth("\u6df1\u5ea6\u89e3\u8bfb\uff1a\u7c89\u4e1d\u7ecf\u6d4e\u5982\u4f55\u6253\u7834", n=2)

if idx2_seat is not None:
    # The seat analysis block should be about 4 paragraphs long
    elems4 = list(body)
    # Find end: next figure paragraph or next section
    seat_end = idx2_seat + 5  # max 5 elements
    for j in range(idx2_seat, min(idx2_seat + 8, len(elems4))):
        el = elems4[j]
        t = get_text(el) if el.tag.endswith("}p") else ""
        if "\u56fe8" in t and ("\u6536\u5165" in t or "\u5ea7\u4f4d" in t):
            seat_end = j + 1
            break
    
    to_remove_seat = elems4[idx2_seat:seat_end]
    print(f"Duplicate seat block: [{idx2_seat}] -> [{seat_end})")
    for el in to_remove_seat:
        body.remove(el)
    print(f"  Removed {len(to_remove_seat)} elements (seat analysis duplicate)")
else:
    print("  Seat analysis duplicate not found or already clean")

# ── Remove duplicate SEM analysis (second occurrence) ──────────────────────
idx2_sem, el2_sem = find_nth("\u56db\u3001\u52a8\u673a-\u963b\u788d\u535a\u5f08\u6a21\u578b\u7684\u6df1\u5ea6\u673a\u5236\u89e3\u8bfb", n=2)

if idx2_sem is not None:
    elems5 = list(body)
    # Find end: look for appendix or next chapter
    sem_end = idx2_sem + 10  # default
    for j in range(idx2_sem, min(idx2_sem + 20, len(elems5))):
        el = elems5[j]
        t = get_text(el) if el.tag.endswith("}p") else ""
        if "\u9644\u5f55" in t or "\u8868C-1" in t:
            sem_end = j
            break
    
    to_remove_sem = elems5[idx2_sem:sem_end]
    print(f"Duplicate SEM block: [{idx2_sem}] -> [{sem_end})")
    for el in to_remove_sem:
        body.remove(el)
    print(f"  Removed {len(to_remove_sem)} elements (SEM analysis duplicate)")
else:
    print("  SEM analysis duplicate not found or already clean")

# ── Save ─────────────────────────────────────────────────────────────────────
doc.save("wfm \u90e8\u5206.docx")
print("\nDone! Duplicates removed and file saved.")

# Verify no more duplicates
doc2 = Document("wfm \u90e8\u5206.docx")
body2 = doc2.element.body

def count_occurrences(keyword):
    count = 0
    for el in list(body2):
        if el.tag.endswith("}p") and keyword in get_text(el):
            count += 1
    return count

kmo_count = count_occurrences("KMO\u53d6\u6837\u9002\u5207\u6027\u68c0\u9a8c\u4e0eBartlett")
seat_count = count_occurrences("\u6df1\u5ea6\u89e3\u8bfb\uff1a\u7c89\u4e1d\u7ecf\u6d4e")
sem_count = count_occurrences("\u56db\u3001\u52a8\u673a-\u963b\u788d\u535a\u5f08\u6a21\u578b\u7684\u6df1\u5ea6")

print(f"\nVerification:")
print(f"  KMO section count: {kmo_count} (should be 1)")
print(f"  Seat analysis count: {seat_count} (should be 1)")
print(f"  SEM analysis count: {sem_count} (should be 1)")
