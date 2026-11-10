#!/usr/bin/env python3
"""The measurements of 70 EPDM 281 out of its verification workbook, as CSV.

    epdm281_extract.py <70EPDM281_verification.xlsx> <out>

Writes into <out> (a new folder; stops if it exists) uniaxial.csv, equibiax.csv, pureshear.csv
(t in s, engineering strain, engineering stress in MPa: the "Measurement" columns) and dma.csv
(frequency in Hz, temperature in C, storage and loss modulus in MPa: the "Messung" columns). The
workbook's simulation columns are left out. The data are not ours to publish: keep <out> out of git
(the paper's data/private, xsesfsi_papers/ldp_containment, is ignored).
"""

import csv
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
SHEETS = {"Uniaxial": "uniaxial", "Equibiax": "equibiax", "Pure shear": "pureshear"}


def numeric_rows(z, target):
    """Per row, the numeric cells by column letter."""
    rows = []
    for r in ET.fromstring(z.read("xl/" + target)).find("m:sheetData", NS).findall("m:row", NS):
        row = {}
        for c in r.findall("m:c", NS):
            v = c.find("m:v", NS)
            if v is not None and c.get("t") != "s":
                row["".join(ch for ch in c.get("r") if ch.isalpha())] = float(v.text)
        rows.append(row)
    return rows


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    book, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        sys.exit("epdm281_extract: %s exists; give a new folder" % out)
    with zipfile.ZipFile(book) as z:
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        sheets = {s.get("name"): rels[s.get(REL)] for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS)}
        tables = {name: numeric_rows(z, sheets[sheet]) for sheet, name in SHEETS.items()}
        dma = numeric_rows(z, sheets["DMA"])
    os.makedirs(out)
    for name, rows in tables.items():
        with open(os.path.join(out, name + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "strain", "stress"])
            for r in rows:
                if all(k in r for k in "ABC"):
                    w.writerow(["%.10g" % r["A"], "%.10g" % r["B"], "%.10g" % r["C"]])
    with open(os.path.join(out, "dma.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frequency", "temperature", "storage", "loss"])
        for r in dma:
            if all(k in r for k in "ABCD"):
                w.writerow(["%.10g" % r[k] for k in "ABCD"])


if __name__ == "__main__":
    main()
