# ============================================================
# scripts/test_entity_profile.py
# Task 7: entity extraction profiles. v1 must reproduce the behaviour of
# the 20 Aug 2026 indexes exactly; v2 must fix each defect found by the
# graph-quality report. Run: python scripts/test_entity_profile.py
# ============================================================
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag.graphrag_indexer import (normalize_entity, is_valid_entity,
                                       set_entity_profile)

FAIL = 0
def check(label, got, want):
    global FAIL
    ok = got == want
    FAIL += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label:55} got={got!r} want={want!r}")

print("v1 (production) - must match the 20 Aug indexes, defects included")
set_entity_profile("v1")
check("analysis -> analysi (legacy bug)", normalize_entity("analysis"), "analysi")
check("databases -> databas (legacy bug)", normalize_entity("databases"), "databas")
check("status -> statu (legacy bug)", normalize_entity("status"), "statu")
check("query analysis rejected (legacy bug)", is_valid_entity(normalize_entity("query analysis")), False)
check("process rejected as generic", is_valid_entity("process"), False)
check("2pc rejected", is_valid_entity("2pc"), False)
check("page range admitted", is_valid_entity("376 382"), True)
check("8086 processor admitted", is_valid_entity("8086 processor"), True)
check("doubled-char artefact admitted", is_valid_entity("rrmmiitt ccllaassssiiffiiccaattiioonn"), True)
check("networks -> network", normalize_entity("networks"), "network")
check("CNNs -> cnn (alias)", normalize_entity("CNNs"), "cnn")

print("\nv2 - fixes")
set_entity_profile("v2")
check("analysis kept", normalize_entity("analysis"), "analysis")
check("query analysis valid", is_valid_entity(normalize_entity("query analysis")), True)
check("databases -> database", normalize_entity("databases"), "database")
check("status kept", normalize_entity("status"), "status")
check("processes -> process", normalize_entity("processes"), "process")
check("queries -> query", normalize_entity("queries"), "query")
check("indexes -> index", normalize_entity("indexes"), "index")
check("zombies -> zombie", normalize_entity("zombies"), "zombie")
check("page tables -> page table (head noun only)", normalize_entity("page tables"), "page table")
check("process valid", is_valid_entity("process"), True)
check("example still rejected", is_valid_entity("example"), False)
check("2pc valid", is_valid_entity("2pc"), True)
check("two-phase commit -> 2pc", normalize_entity("two-phase commit"), "2pc")
check("page range rejected", is_valid_entity("376 382"), False)
check("page-prefixed fragment rejected", is_valid_entity("380 two phase"), False)
check("8086 processor still valid", is_valid_entity("8086 processor"), True)
check("3nf still valid", is_valid_entity("3nf"), True)
check("32 bit still valid", is_valid_entity("32 bit"), True)
check("doubled-char artefact rejected", is_valid_entity("rrmmiitt ccllaassssiiffiiccaattiioonn"), False)
check("'committee' not a doubled-char artefact", is_valid_entity("committee"), True)
check("CNNs -> cnn (alias)", normalize_entity("CNNs"), "cnn")

print("\nv3 - v2 plus fixes found while measuring v2")
set_entity_profile("v3")
check("its variable -> variable", normalize_entity("its variable"), "variable")
check("each signal -> signal", normalize_entity("each signal"), "signal")
check("my availability -> availability", normalize_entity("my availability"), "availability")
check("two fork -> fork", normalize_entity("two forks"), "fork")
check("no preemption kept (deadlock condition)", normalize_entity("no preemption"), "no preemption")
check("first fit kept", normalize_entity("first fit"), "first fit")
check("most significant bit kept", normalize_entity("most significant bit"), "most significant bit")
check("two-phase commit -> 2pc (not 'phase commit')", normalize_entity("two-phase commit"), "2pc")
check("three tier architecture kept", normalize_entity("three tier architecture"), "three tier architecture")
check("number rejected again", is_valid_entity("number"), False)
check("process still valid", is_valid_entity("process"), True)
check("3nf valid", is_valid_entity("3nf"), True)
check("analysis kept", normalize_entity("analysis"), "analysis")

print("\nretriever neighbour weighting (toy graph)")
import networkx as nx
from graphrag.graphrag_retriever import _neighbour_score
g = nx.Graph(total_chunks=1000)
g.add_node("deadlock", doc_freq=100); g.add_node("time", doc_freq=600)
g.add_node("avoidance", doc_freq=20); g.add_node("its frag", doc_freq=2)
g.add_edge("deadlock", "time", weight=15); g.add_edge("deadlock", "avoidance", weight=11)
g.add_edge("deadlock", "its frag", weight=2)
for n in range(300): g.add_edge("time", f"x{n}", weight=2)   # time is a hub
def top(w): return sorted(g["deadlock"], key=lambda v: -_neighbour_score(g, "deadlock", v, 0.5, w))[0]
check("npmi_logcount prefers the specific concept", top("npmi_logcount"), "avoidance")
check("missing total_chunks falls back safely", _neighbour_score(nx.Graph([("a","b",{"weight":3})]), "a", "b", 0.5, "npmi_logcount") > 0, True)

set_entity_profile("v1")
print(f"\n{'ALL PASSED' if not FAIL else f'{FAIL} FAILED'}")
sys.exit(1 if FAIL else 0)
