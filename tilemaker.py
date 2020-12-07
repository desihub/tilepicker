
from numpy.lib import recfunctions
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def make_tiles_from_fiberassign(dirname, gaiadensitymapfile, 
                                tycho2file, covfile):
    fn = glob.glob(os.path.join(dirname, '**/fiberassign*.fits.gz'),
                   recursive=True)
    tiles = numpy.zeros(len(fn), dtype=basetiledtype)
    for i, fn0 in enumerate(fn):
        h = fits.getheader(fn0)
        tiles['tileid'][i] = h['TILEID']
        tiles['ra'][i] = h['TILERA']
        tiles['dec'][i] = h['TILEDEC']
        tiles['program'][i] = h['FA_SURV'].strip()+'_'+h['FLAVOR'].strip()
    tiles['airmass'] = airmass(
        numpy.ones(len(tiles), dtype='f4')*15., tiles['dec'], 31.96)
    tiles_add = add_info_fields(tiles, gaiadensitymapfile, 
                                tycho2file, covfile)
    tiles = recfunctions.merge_arrays((tiles, tiles_add), flatten=True)
    signalfac = 10.**(3.303*tiles['ebv_med']/2.5)
    tiles['exposefac'] = signalfac**2 * tiles['airmass']**1.25
    tiles['centerid'] = tiles['tileid']
    for i, prog in enumerate(numpy.unique(tiles['program'])):
        m = tiles['program'] == prog
        tiles['pass'][m] = i
    coord = SkyCoord(ra=tiles['ra']*u.deg, 
                     dec=tiles['dec']*u.deg, frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    dt = tiles['dec']
    tiles['in_desi'] = (
        (tiles['in_imaging'] != 0) & (dt >= -18) & (dt <= 77.7) & 
        ((bt > 0) | (dt < 32.2)) &
        (((numpy.abs(bt) > 22) & ((lt < 90) | (lt > 270))) | 
         ((numpy.abs(bt) > 20) & (lt > 90) & (lt < 270))))
    tiles['obsconditions'] = 2**31-1
    return tiles

if __name__ == '__main__':

    p = ArgumentParser(description='DESI tile creator.')
    p.add_argument('input', nargs='+',
                   help='Fiberassign FITS file(s)')
    p.add_argument('-c', '--covfile', required=True,
                   help='Coverage file.')
    p.add_argument('-g', '--gaiafile', required=True,
                   help='Gaia density map file.')
    p.add_argument('-t', '--tycho2file', required=True,
                   help='Tycho-2 file.')

    args = p.parse_args()

