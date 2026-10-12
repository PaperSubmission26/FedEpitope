import requests
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT = Path("data/ncbi_ids.txt")
OUTPUT = Path("data/ncbi_proteins.fasta")
FAILED = Path("logs/ncbi_failed.txt")
MAPFILE = Path("data/ncbi_sequence_map.tsv")

# Conservative number of simultaneous NCBI requests
MAX_WORKERS = 3

# Number of times to retry when NCBI rate-limits us
MAX_RETRIES = 5


ids = [
    x.strip()
    for x in INPUT.read_text().splitlines()
    if x.strip()
]

# PDB-style identifiers will be handled separately
pdb_pattern = re.compile(r'^[0-9][A-Za-z0-9]{3}_[0-9A-Za-z]+$')

ids_to_fetch = [x for x in ids if not pdb_pattern.fullmatch(x)]
pdb_ids = [x for x in ids if pdb_pattern.fullmatch(x)]

print("Total NCBI IDs:", len(ids))
print("Trying through NCBI:", len(ids_to_fetch))
print("PDB-like skipped:", len(pdb_ids))
print("Parallel workers:", MAX_WORKERS)
print("Max 429 retries:", MAX_RETRIES)


def fetch(acc):
    """
    Fetch one protein sequence from NCBI.

    HTTP 429 responses are retried using exponential backoff.
    """

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=protein&id={acc}&rettype=fasta&retmode=text"
    )

    for attempt in range(MAX_RETRIES + 1):

        try:
            r = requests.get(
                url,
                headers={
                    "User-Agent": "FedEpitope-revision/1.0"
                },
                timeout=30
            )

            text = r.text.strip()

            # Successful FASTA response
            if (
                r.status_code == 200
                and text.startswith(">")
                and "\n" in text
            ):
                return acc, text, None

            # NCBI rate limiting
            if r.status_code == 429:

                if attempt < MAX_RETRIES:

                    # 2, 4, 8, 16, 32 seconds
                    wait_time = 2 ** (attempt + 1)

                    print(
                        f"429 for {acc}; "
                        f"retry {attempt + 1}/{MAX_RETRIES} "
                        f"after {wait_time}s"
                    )

                    time.sleep(wait_time)
                    continue

                return acc, None, "HTTP_429_AFTER_RETRIES"

            # Other HTTP error
            return acc, None, f"HTTP_{r.status_code}"

        except Exception as e:

            # Retry network/time-out errors as well
            if attempt < MAX_RETRIES:

                wait_time = 2 ** (attempt + 1)

                print(
                    f"Error for {acc}: {e}; "
                    f"retry {attempt + 1}/{MAX_RETRIES} "
                    f"after {wait_time}s"
                )

                time.sleep(wait_time)
                continue

            return acc, None, str(e)

    return acc, None, "UNKNOWN_ERROR"


success = 0
failed = 0

with open(OUTPUT, "w") as out, \
     open(FAILED, "w") as fail, \
     open(MAPFILE, "w") as mapping:

    mapping.write("requested_id\tfasta_header\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(fetch, acc): acc
            for acc in ids_to_fetch
        }

        for i, future in enumerate(as_completed(futures), start=1):

            acc, fasta, error = future.result()

            if fasta:

                lines = fasta.splitlines()
                original_header = lines[0]

                # Replace NCBI header with our original identifier
                out.write(f">NCBI|{acc}\n")
                out.write("".join(lines[1:]) + "\n")

                # Keep original NCBI FASTA header for traceability
                mapping.write(
                    f"{acc}\t{original_header.lstrip('>')}\n"
                )

                success += 1

            else:

                fail.write(f"{acc}\t{error}\n")
                failed += 1

            # Make files update immediately
            out.flush()
            fail.flush()
            mapping.flush()

            if i % 25 == 0:
                print(
                    f"Processed {i}/{len(ids_to_fetch)} "
                    f"| Successful: {success} "
                    f"| Failed: {failed}"
                )


# Record PDB entries separately
with open(FAILED, "a") as fail:

    for acc in pdb_ids:
        fail.write(
            f"{acc}\tPDB_LIKE_SKIPPED\n"
        )


print("\nDone.")
print("Successful:", success)
print("Failed:", failed)
print("PDB-like skipped:", len(pdb_ids))
print("FASTA:", OUTPUT)
print("Failures:", FAILED)
print("Mapping:", MAPFILE)
