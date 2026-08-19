import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

print("=" * 60)
print("  RUNNING FULL-CYCLE VERIFICATION TEST SUITE")
print("=" * 60)

# 1. HTML Pages
for page in ["/", "/home.html", "/chat.html", "/uploaded-pdfs.html", "/citation-history.html"]:
    res = client.get(page)
    assert res.status_code == 200, f"Failed to load {page}: {res.status_code}"
    print(f"[OK] Static HTML Page {page:25} -> 200 OK")

# 2. Assets / Static Files
for asset in ["/js/chat.js", "/css/chat.css", "/js/home.js", "/js/library.js", "/js/history.js"]:
    res = client.get(asset)
    assert res.status_code == 200, f"Failed to load {asset}: {res.status_code}"
    print(f"[OK] Static Asset     {asset:25} -> 200 OK")

# 3. Health & Sources APIs
res = client.get("/api/health")
assert res.status_code == 200 and res.json().get("status") == "healthy"
print(f"[OK] API /api/health            -> 200 OK (Status: {res.json()['status']})")

res = client.get("/api/sources")
assert res.status_code == 200 and len(res.json().get("sources", [])) == 3
print(f"[OK] API /api/sources           -> 200 OK ({len(res.json()['sources'])} connected guidelines: NG101, CG81, CG164)")

# 4. PDF Endpoints
for pdf in ["NG101.pdf", "CG81.pdf", "CG164.pdf"]:
    res = client.get(f"/api/pdf/{pdf}")
    assert res.status_code == 200 and len(res.content) > 10000
    print(f"[OK] PDF Serving /api/pdf/{pdf:12} -> 200 OK ({len(res.content):,} bytes)")

# 5. Clinical Query Grounded RAG (Early Breast Cancer - NG101)
print("\n[*] Testing Clinical Query (NG101): 'What is the recommended endocrine therapy for hormone receptor positive early breast cancer?'")
res = client.post("/api/chat", json={
    "question": "What is the recommended endocrine therapy for hormone receptor positive early breast cancer?"
})
assert res.status_code == 200
data = res.json()
assert data.get("success") is True
assert data.get("has_context") is True
assert len(data.get("recommendations", [])) > 0
assert len(data.get("citations", [])) > 0
print(f"[OK] Clinical Query RAG Response -> 200 OK")
print(f"    Confidence: {data.get('confidence')} (Source Match: {data.get('source_match')})")
print(f"    Summary: {data.get('summary')[:90]}...")
print(f"    Recommendations: {len(data.get('recommendations'))} points")
print(f"    Supporting Evidence: {len(data.get('supporting_evidence'))} points")
print(f"    Retrieved Citations: {len(data.get('citations'))} chunks:")
for c in data.get("citations", []):
    print(f"      - {c['source']} {c['section']} ({c['pages']}) | Score: {c['hybrid_score']}")

# 6. Clinical Query Grounded RAG (Familial Breast Cancer - CG164)
print("\n[*] Testing Clinical Query (CG164): 'What surveillance is recommended for BRCA mutation carriers?'")
res = client.post("/api/chat", json={
    "question": "What surveillance is recommended for BRCA mutation carriers?"
})
assert res.status_code == 200
data = res.json()
assert data.get("success") is True
assert data.get("has_context") is True
assert len(data.get("recommendations", [])) > 0
assert len(data.get("citations", [])) > 0
print(f"[OK] Familial Breast Cancer RAG Response -> 200 OK")
print(f"    Confidence: {data.get('confidence')} (Source Match: {data.get('source_match')})")
print(f"    Summary: {data.get('summary')[:90]}...")
print(f"    Retrieved Citations: {len(data.get('citations'))} chunks:")
for c in data.get("citations", []):
    print(f"      - {c['source']} {c['section']} ({c['pages']}) | Score: {c['hybrid_score']}")

# 7. Unrelated / Out-of-Context Query (Hallucination Rejection)
print("\n[*] Testing Out-of-Context Query (Threshold Rejection): 'What is the capital of Australia?'")
res = client.post("/api/chat", json={
    "question": "What is the capital of Australia?"
})
assert res.status_code == 200
data = res.json()
assert data.get("has_context") is False
assert data.get("confidence") == "Low"
assert len(data.get("citations")) == 0
print(f"[OK] Out-of-Context Rejection   -> 200 OK (Confidence: Low, 0 hallucinations)")

print("\n" + "=" * 60)
print("  ALL FULL-CYCLE VERIFICATION TESTS PASSED SUCCESSFULLY!  ")
print("=" * 60)
