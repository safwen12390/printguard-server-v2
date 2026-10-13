from dotenv import load_dotenv
load_dotenv()
from ai_service import retrieve_for_defect

TEST_QUERIES = {
    "spaghetti": [
        "why did my print turn into spaghetti",
        "filament ball on the print bed",
        "print detached mid-print",
    ],
    "warping": [
        "corners lifting from the bed",
        "warping on large flat parts",
        "bed temperature causing warp",
    ],
    "under_extrusion": [
        "missing material in layers",
        "gaps in the print walls",
        "clogged nozzle under-extrusion",
    ],
    "stringing": [
        "hairs between printed parts",
        "oozing during travel moves",
        "retraction settings for stringing",
    ],
}

for cls, queries in TEST_QUERIES.items():
    print(f"\n{'=' * 70}")
    print(f"CLASS: {cls}")
    print('=' * 70)
    for q in queries:
        docs = retrieve_for_defect(cls, k=3, user_question=q)
        print(f"\nQ: {q}")
        for d in docs:
            print(f"  [{d['doc_id']:<22}] d={d['distance']:.3f}  {d['title'][:55]}")