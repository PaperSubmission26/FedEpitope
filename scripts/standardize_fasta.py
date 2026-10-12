from pathlib import Path

UNIPROT = Path("data/uniprot_proteins.fasta")
NCBI = Path("data/ncbi_proteins.fasta")
OUTPUT = Path("data/all_proteins.fasta")


def write_uniprot(out):
    with open(UNIPROT) as f:
        header = None
        seq = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    out.write(f">{header}\n")
                    out.write("".join(seq) + "\n")

                parts = line.split("|")

                if len(parts) >= 2:
                    acc = parts[1]
                else:
                    acc = line[1:].split()[0]

                header = f"UNIPROT|{acc}"
                seq = []

            else:
                seq.append(line)

        if header is not None:
            out.write(f">{header}\n")
            out.write("".join(seq) + "\n")


def write_ncbi(out):
    with open(NCBI) as f:
        header = None
        seq = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    out.write(f">{header}\n")
                    out.write("".join(seq) + "\n")

                # Already stored as >NCBI|accession
                header = line[1:]
                seq = []

            else:
                seq.append(line)

        if header is not None:
            out.write(f">{header}\n")
            out.write("".join(seq) + "\n")


with open(OUTPUT, "w") as out:
    write_uniprot(out)
    write_ncbi(out)

print("Saved:", OUTPUT)