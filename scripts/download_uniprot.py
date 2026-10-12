import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT = Path("data/uniprot_ids.txt")
OUTPUT = Path("data/uniprot_proteins.fasta")
FAILED = Path("logs/uniprot_failed.txt")

ids = [x.strip() for x in INPUT.read_text().splitlines() if x.strip()]

print("Total UniProt IDs:", len(ids))


def fetch_protein(acc):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"

    try:
        r = requests.get(
            url,
            headers={"User-Agent": "FedEpitope-revision/1.0"},
            timeout=30
        )

        if r.status_code == 200 and r.text.startswith(">"):
            return acc, r.text, None

        return acc, None, f"HTTP_{r.status_code}"

    except Exception as e:
        return acc, None, str(e)


with open(OUTPUT, "w") as out, open(FAILED, "w") as fail:

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {
            executor.submit(fetch_protein, acc): acc
            for acc in ids
        }

        for i, future in enumerate(as_completed(futures), start=1):

            acc, fasta, error = future.result()

            if fasta:
                out.write(fasta)
                if not fasta.endswith("\n"):
                    out.write("\n")
            else:
                fail.write(f"{acc}\t{error}\n")

            if i % 100 == 0:
                print(f"Processed {i}/{len(ids)}")

print("Done.")
print("FASTA:", OUTPUT)
print("Failed IDs:", FAILED)