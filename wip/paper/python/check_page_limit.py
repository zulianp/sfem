#!/usr/bin/env python3
"""Check the page containing the last body text, independently of PDF column order."""
import argparse
from pathlib import Path
import re
import subprocess


def body_page(aux):
    pages = re.findall(r'\\newlabel\{pasc-body-end\}\{\{[^{}]*\}\{([0-9]+)\}', aux)
    if len(pages) != 1 or int(pages[0]) < 1:
        raise ValueError('missing or duplicate end-of-body page label; rebuild the paper')
    return int(pages[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('page limit must be positive')
    source = Path('main.tex').read_text()
    acknowledgments = re.findall(r'\\begin\{acks\}(.*?)\\end\{acks\}', source, re.S)
    if (len(acknowledgments) != 1 or
            acknowledgments[0].count(r'\label{pasc-body-end}') != 1 or
            source.count(r'\label{pasc-body-end}') != 1):
        raise ValueError('the end-of-body label must occur once in the acknowledgments')
    body = body_page(Path('main.aux').read_text())
    info = subprocess.check_output(['pdfinfo', 'main.pdf'], universal_newlines=True)
    total = int(re.search(r'^Pages:\s+(\d+)\s*$', info, re.M).group(1))
    if body > total:
        raise ValueError('body label lies beyond the PDF; rebuild the paper')
    print('{} body pages; {} total pages (limit {}, excluding references)'.format(body, total, args.limit))
    if body > args.limit:
        raise SystemExit('OVER THE LIMIT -- PASC grants no extensions')


if __name__ == '__main__':
    main()
