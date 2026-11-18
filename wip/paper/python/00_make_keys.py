#!/usr/bin/env python3
# File       : 00_make_keys.py
# Created    : Tue Nov 17 2026 20:16:33 (+0100)
# Author     : Fabian Wermelinger
# Description: Extraction script for figure legend keys

from paper_tools import mpl_tools
import matplotlib.pyplot as plt

plt.style.use('pasc27.mplstyle')

cycl = mpl_tools.getCycler()
mpl_tools.exportKeys(cycl['linestyle'], cycl['marker'], cycl['color'])
