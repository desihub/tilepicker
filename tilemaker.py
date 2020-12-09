"""tilemaker.py

Generate a DESI tile file from a list of fiberassign files.
"""

import numpy as np
from numpy.lib import recfunctions

import fitsio
from astropy.coordinates import SkyCoord
from astropy import units as u

from desisurvey.tileqa import make_tiles_from_fiberassign

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == '__main__':

    p = ArgumentParser(description='DESI tile creator.')
    p.add_argument('-c', '--covfile', required=True,
                   help='Coverage file.')
    p.add_argument('-g', '--gaiadensitymapfile', required=True,
                   help='Gaia density map file.')
    p.add_argument('-t', '--tycho2file', required=True,
                   help='Tycho-2 file.')
    p.add_argument('-d', '--dirname', required=True,
                   help='Fiberassign FITS files folder.')
    p.add_argument('-o', '--output', required=True,
                   help='FITS output with DESI tile info.')

    args = p.parse_args()
    output = fitsio.FITS(args.output, 'rw', clobber=True)
    del args.output

    tiles = make_tiles_from_fiberassign(**vars(args))
    tiles.dtype.names = [_x.upper() for _x in tiles.dtype.names]

    output.write(tiles)
