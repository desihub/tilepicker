import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font', size=13)

from datetime import date, datetime, timezone, timedelta

import ephem

from astropy.io import ascii
from astropy.coordinates import get_moon, get_sun
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.visualization import astropy_mpl_style, quantity_support
from astropy.time import Time
import astropy.units as u

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def altToAirmass(a):
    return 1./np.cos(np.radians(90. - a))

def airmassToAlt(a):
    a[a < 1] = 1.
    return 90. - np.degrees(np.arccos(1./a))

def twilight(delta_t, sunaltaz, sunangle=-18*u.deg):
    n = len(sunaltaz)
    t0 = np.interp(sunangle, sunaltaz.alt[:n//2], delta_t[:n//2], period=180.)
    t1 = np.interp(sunangle, sunaltaz.alt[n//2:], delta_t[n//2:], period=180.)
    return t0, t1

def deltaToHr(d):
    return (d + 24) % 24

def hrToDelta(hr):
    d = hr - 24
    d[d < -12] += 24
    return d

if __name__ == '__main__':
    p = ArgumentParser(description='Transit alt/az plotter.')
    p.add_argument('infile', nargs=1,
                   help='File with tile list. CSV format: TILE,RA,DEC')
    p.add_argument('-o', '--output', default='tile_altaz.png',
                   help='Output filename [PNG format]')
    args = p.parse_args()

    g = ephem.Observer()
    g.name='Somewhere'
    g.lat=np.radians(31.9)  # lat/long in decimal degrees
    g.long=np.radians(-111.6)
    m = ephem.Moon()
    tz = timezone(timedelta(hours=-7))
    g.date = datetime.now().astimezone(tz)
    m.compute(g)

    kpno = EarthLocation(lat=31.9*u.deg, lon=-111.6*u.deg, height=2100*u.m)
    utcoffset = -7*u.hour  # Mountain Standard Time [US/Arizona]

    midnight = Time('{} 00:00:00'.format(date.today())) - utcoffset
    delta_midnight = np.linspace(-7, 7, 250)*u.hour
    times = midnight + delta_midnight
    frame = AltAz(obstime=times, location=kpno)

    sep_hours = np.linspace(-6,6,7)*u.hour
    sep_idx = [np.abs(delta_midnight-h).argmin() for h in sep_hours]

    moon = get_moon(times, location=kpno)
    moonaltaz = moon.transform_to(frame)

    sun = get_sun(times)
    sunaltaz = sun.transform_to(frame)

    fig, ax = plt.subplots(1,1, figsize=(10,7))

    ax.plot(delta_midnight, moonaltaz.alt, ls='--', lw=4,
            alpha=0.7*m.phase/100 + 0.3,
            label='Moon (phase {:.0f}%)'.format(m.phase))

    data = ascii.read(args.infile[0], format='csv')
    print(data)
    n = len(data)
    n = 5
    col = mpl.cm.magma(np.linspace(0.1,0.75,n))
    lin = ['-', '-.', '--', ':']

    for i, (_tile_id, _ra, _dec) in enumerate(data):
        print('{} {} {} '.format(_tile_id, _ra, _dec))
        tile_radec = SkyCoord(ra=_ra*u.deg, dec=_dec*u.deg, frame='icrs')

        tile_altaz = tile_radec.transform_to(frame)
        ax.plot(delta_midnight, tile_altaz.alt, lw=2, alpha=0.7,
                color=col[i%n],
                ls=lin[i//n % len(lin)],
                label='{}'.format(_tile_id))

        for j in sep_idx:
            sep = moon[j].separation(tile_radec).degree

            if tile_altaz[j].alt.degree >= 0 and tile_altaz[j].alt.degree < 85:
                ax.text(x=delta_midnight[j].value, y=tile_altaz[j].alt.degree + 2,
                        s=r'{:.0f}$\degree$'.format(sep), fontsize=7,
                        horizontalalignment='center',
                        verticalalignment='center',
                        color=col[i%n])

    ax.legend(loc='upper center', ncol=3,
              bbox_to_anchor=(0.5, 1.15),
              fancybox=True, shadow=True,
              fontsize=9)

    # Draw times before twilight ends / after twilight begins.
    t0, t1 = twilight(delta_midnight, sunaltaz)
    ax.fill_between([-7, t0], 0, 90, color='r', alpha=0.1)
    ax.fill_between([t1, 7], 0, 91, color='r', alpha=0.1)
    ax.text(t0, 91, r'$-18\degree$', horizontalalignment='right', color='r', fontsize=6, fontweight='heavy')
    ax.text(t1, 91, r'$-18\degree$', horizontalalignment='left', color='r', fontsize=6, fontweight='heavy')

    t0, t1 = twilight(delta_midnight, sunaltaz, sunangle=-12*u.deg)
    ax.fill_between([-7, t0], 0, 91, color='r', alpha=0.1)
    ax.fill_between([t1, 7], 0, 91, color='r', alpha=0.1)
    ax.text(t0, 91, r'$-12\degree$', horizontalalignment='right', color='r', fontsize=6, fontweight='heavy')
    ax.text(t1, 91, r'$-12\degree$', horizontalalignment='left', color='r', fontsize=6, fontweight='heavy')

    # Gray out airmass > 2.
    ax.fill_between([-7, 7], 0, 30, color='k', alpha=0.1)

    # Draw midnight + current time.
    ax.axvline(0, color='k', ls='--', lw=1)

    # Draw current time.
    tnow = (Time(datetime.now().astimezone(tz)) - midnight).to('hour').value
    ax.axvline(tnow, color='k', ls=':')

    def deltaToHr(d):
        return (d + 24) % 24

    deltas = np.arange(-7,8)
    hrs = deltaToHr(deltas)
    hr_labels = ['{}h'.format(_hr) for _hr in hrs]

    ax.set(xlim=(-7, 7),
           xticks=deltas,
           xticklabels=hr_labels,
           ylim=(0, 90),
           yticks=np.arange(0,90+10,10),
           xlabel='time [MST - US/Arizona]',
           ylabel='altitude [deg]',
           #title='{}'.format(midnight)
           )
    ax.grid(ls=':')

    ax_y2 = ax.secondary_yaxis('right', functions=(altToAirmass, airmassToAlt))
    ax_y2.set_yticks([1,1.2,1.5,2,3,5,1000])
    ax_y2.set_yticklabels(['1','1.2','1.5','2','3','5','38'])
    ax_y2.set_ylabel(r'airmass')

    #ax_x2 = ax.secondary_xaxis('top', functions=(deltaToHr, hrToDelta))
    #hrs = deltaToHr(np.arange(-7,8))
    #hr_labels = ['{}h'.format(_hr) for _hr in hrs]
    #ax_x2.set(xticks=hrs,
    #          xticklabels=hr_labels,
    #          xlabel=r'time [US/Arizona - MST]')

    #fig.tight_layout()
    #fig.subplots_adjust(right=0.92)

    fig.savefig(args.output, dpi=120)
    plt.show()
