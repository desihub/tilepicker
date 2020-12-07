import sys,os,math,time,json,fitsio
import os.path
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pylab as py

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, ICRS, EarthLocation


from bokeh.plotting import figure, show

from bokeh.embed import components
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Range1d, Label, TapTool, OpenURL, CustomJS, CrosshairTool, LinearAxis
from bokeh.io import output_notebook
from bokeh.models.widgets import RadioButtonGroup
from bokeh.layouts import column
from bokeh.models import CustomJS
from bokeh.models.glyphs import Text
from bokeh.transform import linear_cmap
from bokeh.models.widgets import tables as bktables
from bokeh.models import CustomJS, ColumnDataSource, DateSlider, DateRangeSlider
from datetime import datetime as dt
from datetime import datetime
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.coordinates import SkyCoord, EarthLocation, AltAz,solar_system_ephemeris,get_body,get_body_barycentric

from astropy.time import Time, TimezoneInfo
from bokeh.embed import file_html
from bokeh.resources import CDN
from astropy.coordinates import get_moon

import pylunar
from astroplan import Observer

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import ephem
from pytz import timezone
from datetime import datetime


class Observatory:
    def __init__(self, name='KPNO-Mayall', lon=-111.6*u.degree, lat=31.9633*u.degree, elev=2120.*u.meter, tzname='US/Arizona'):
        """Initialize observatory data with KPNO-4m values as the default.

        Parameters
        ----------
        name : str
            Observatory name, e.g., 'KPNO', 'Kitt Peak', etc.
        lon : float
            Observatory longitude, in degrees.
        lat : float
            Observatory latitude, in degrees.
        elev : float
            Observatory elevation above sea level, in meters.
        tzname : str
            Name of observatory time zone.
        tzoffset : int
            Time offset from UTC, in hours.
        """

        self.name = name
        self.obs = EarthLocation(lon=lon, lat=lat, height=elev)
        self.tzone = timezone('US/Arizona')

        # Create a PyEphem observatory object for tracking the Moon.
        self._ephem_obs = ephem.Observer()
        self._ephem_obs.lon = '{}'.format(lon.to('degree').value)
        self._ephem_obs.lat = '{}'.format(lat.to('degree').value)
        self._ephem_obs.elevation = elev.to('m').value
        self._ephem_obs.horizon = '-18'

    def to_local_datetime(self, date):
        """Convert a datetime object to local date and time for this
        observatory.

        Parameters
        ----------
        date : str or datetime
            Date + time, in format YYYY/MM/DD HH:MM:SS if str.

        Returns
        -------
        datestr : str
            Date in format YYYY/MM/DD HH:MM:SS in current time zone.
        """
        if type(date) == datetime:
            date_str = date.astimezone(self.tzone).strftime('%Y/%m/%d %H:%M:%S')
        else:
            date_str = date
        return date_str

    def get_moon(self, obstime):
        """Get topocentric moon location and phase.

        Parameters
        ----------
        time : astropy.Time
            Date + time.

        Returns
        -------
        ra : float
            Topocentric RA of Moon for this observatory.
        dec : float
            Topocentric Dec of Moon for this observatory.
        phase : float
            Illuminated fraction of the Moon, between 0 and 1.
        """
        self._ephem_obs.date = obstime.to_datetime().astimezone(self.tzone)
        moon = ephem.Moon(self._ephem_obs)
        return np.degrees([moon.ra, moon.dec]).tolist() + [moon.moon_phase]

    def get_twilight(self, obstime):
        """Compute start and end of astro-twilight.

        Parameters
        ----------
        obstime : astropy.Time
            Observation time.

        Returns
        -------
        """
        self._ephem_obs.date = obstime.to_datetime()

        # Begin and end astronomical twilight.
        tbeg = self._ephem_obs.previous_rising(ephem.Sun(), use_center=True)
        tend = self._ephem_obs.next_setting(ephem.Sun(), use_center=True)

        return str(tbeg), str(tend)

    def airmass_limit_range(self, airmass, obstime):
        """Return observatory 'horizon' for a given airmass at a given time.

        Parameters
        ----------
        airmass : float
            Limiting airmass (e.g. 1.5, 2.0, etc.)
        obstime : astropy.Time
            Observation time.

        Returns
        -------
        ra : ndarray
            List of RA values defining the current limiting airmass.
        dec : ndarray
            List of Dec values defining the current limiting airmass.
        """
        zenith = np.degrees(np.arccos(1./airmass))
        az = np.arange(0, 360+3, 3)*u.degree
        alt = np.full(az.shape, 90.-zenith)*u.degree
        s = AltAz(alt=alt, az=az, obstime=obstime, location=self.obs)
        c = s.transform_to(ICRS)

        return c.ra.to('degree').value, c.dec.to('degree').value

    def __str__(self):
        """String representation of the observatory.
        """
        return '\n'.join([self.name,
                          'lon:  {:.2f}'.format(self.obs.lon),
                          'lat:  {:.4f}'.format(self.obs.lat),
                          'elev: {:.2f}'.format(self.obs.height),
                          'TZ:   {}'.format(self.tzone)])

observer = Observatory()


def html_header(title):
    """Output tilepicker HTML header.

    Parameters
    ----------
    title : str
        Page title.

    Returns
    -------
    head : str
        HTML topmatter.
    """
    head = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '  <head>',
        '    <meta charset="utf-8">',
        '    <title>{}</title>'.format(title),
        '      <link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.13.0.min.css" type="text/css" />',
        '      <link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.13.0.min.css" type="text/css" />',
        '      <link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-tables-0.13.0.min.css" type="text/css" />',
        '      <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.13.0.min.js"></script>',
        '      <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.13.0.min.js"></script>',
        '      <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-tables-0.13.0.min.js"></script>',
        '      <script type="text/javascript">',
        '          Bokeh.set_log_level("info");',
        '      </script>',
        '  </head>',
        '  <body>']
    
    return '\n'.join(head)


def html_footer():
    """Output tilepicker HTML closing tags.

    Returns
    -------
    tail : str
        HTML body and page closing tags.
    """
    
    ## HTML footer
    tail = [
        '  </body>',
        '</html>']

    return '\n'.join(tail)


#########################################################
def get_kp_twilights(tt, dd):  # example: '2020-01-01 12:00:00'

    kp = Observer.at_site("Kitt Peak", timezone="MST")
    ds = str(dd[0]) + '-' + str(dd[1]) + '-' + str(dd[2])
    ts = str(tt[0]) + ':' + str(tt[1]) + ':' + str(tt[2])
    t = Time(ds + ' ' + ts)

    eve = kp.twilight_evening_astronomical(t, which='next').datetime
    mor = kp.twilight_morning_astronomical(t, which='next').datetime

    return eve, mor


#########################################################
def moonLoc(tt, dd, loc, mooninfo_obj):
    
    ds = str(dd[0]) + '-' + str(dd[1]) + '-' + str(dd[2])
    ts = str(tt[0]) + ':' + str(tt[1]) + ':' + str(tt[2])
    t = Time(ds + ' ' + ts)

    # loc = EarthLocation.of_site('Kitt Peak')
    moon_loc = get_moon(t, loc)
    moon_update = mooninfo_obj.update(str(t))
    frac_phase = mooninfo_obj.fractional_phase()
    name_phase = mooninfo_obj.phase_name()
    return moon_loc.ra, moon_loc.dec, frac_phase, name_phase


#########################################################

def jupLoc(tt, dd, loc):
    ds = str(dd[0]) + '-' + str(dd[1]) + '-' + str(dd[2])
    ts = str(tt[0]) + ':' + str(tt[1]) + ':' + str(tt[2])
    t = Time(ds + ' ' + ts)
    # loc = EarthLocation.of_site('Kitt Peak')
    with solar_system_ephemeris.set('builtin'):
        jup_loc = get_body('jupiter', t, loc)

    return jup_loc.ra, jup_loc.dec


def add_plane(p, color='black', plane=None, projection=None):
    """Add a plane (ecliptic, galactic, etc.), to the plot.

    Parameters
    ----------
    p : bokeh.plotting.figure
        Figure object (passed by reference).
    color : str
        Color of the plane line.
    plane : str
        Name of plane coordinate system (astropy convention required).
    projection : str
        Name of target coordinate system (astropy convention required).
    """
    if plane == None or projection == None:
        return

    lon = np.arange(0., 360, 2)
    lat = np.zeros_like(lon)

    c1 = SkyCoord(lon, lat, unit='degree', frame=plane.lower())
    c2 = c1.transform_to(projection.lower())
    alpha, delta = c2.ra.value, c2.dec.value

    # for i in range(len(alpha)):
    #  if alpha[i] >180:
    ind = np.argsort(alpha)
    alpha = alpha[ind]
    delta = delta[ind]

    p.line(alpha, delta, line_width=2, color=color)
    p.line(alpha + 5, delta + 5, color='black', alpha=0.5)
    p.line(alpha - 5, delta - 5, color='black', alpha=0.5)


def calc_airmass_limit(tt, dd, airmass_lim):
    """Compute a circle around the observatory at some limiting airmass.

    Parameters
    ----------
    tt : list of int
        Time [hr, min, sec].
    dd : list of int
        Date [yyyy, mm, dy]
    airmass_lim : float
        Limiting airmass for the calculation.

    Returns
    -------
    circle : ColumnDataSource
        Coordinates indicating limiting airmass above the observatory.
    """
    ds = '-'.join([str(_x) for _x in dd])
    ts = ':'.join([str(_x) for _x in tt])

    obs_time = observer.to_local_time(datetime(*dd, *tt))
    reference = datetime(2000,1,1,12,0,0) # UTC



#########################################################
def skyCircle(tt, dd, airmass_lim):
    ds = str(dd[0]) + '-' + str(dd[1]) + '-' + str(dd[2])
    ts = str(tt[0]) + ':' + str(tt[1]) + ':' + str(tt[2])

    observatory = "kpno"
    name = "Kitt Peak National Observatory"
    Lon = -(111 + 35 / 60. + 59.6 / 3600)
    Lat = 31.9599
    Height = 2120.  # meter

    obsTime = dt(dd[0], dd[1], dd[2], tt[0], tt[1], tt[2])
    reference = dt(2000, 1, 1, 12, 0, 0)  # UTCref
    reference = time.mktime(reference.timetuple())
    obsTime = time.mktime(obsTime.timetuple()) + 7 * 3600
    deltaT = (obsTime - reference) / (24 * 3600);

    # Convert to LST
    LST_hours = ((18.697374558 + 24.06570982441908 * deltaT) + Lon / 15.) % 24;
    LST_degrees = LST_hours * 15

    obsTime = Time(ds + ' ' + ts)
    loc = EarthLocation(lat=Lat * u.deg, lon=Lon * u.deg, height=Height * u.m)

    zenithangle = math.acos(
        1 / airmass_lim) * 180 / math.pi  # 48.19  # deg The zenith angle at which airmass equals 1.5

    az = np.arange(0, 360, 3)
    alt = az * 0 + (90 - zenithangle)

    newAltAzcoordiantes = SkyCoord(alt=alt, az=az, obstime=obsTime, frame='altaz', location=loc, unit="deg")
    ra = newAltAzcoordiantes.icrs.ra.deg
    dec = newAltAzcoordiantes.icrs.dec.deg

    newAltAzcoordiantes = SkyCoord(alt=[90], az=[90], obstime=obsTime, frame='altaz', location=loc, unit="deg")
    ra0 = newAltAzcoordiantes.icrs.ra.deg
    ra0 = ra0[0]
    ra0 = (ra - ra0) % 360

    return ColumnDataSource({"RA0": ra0, "RA": (ra0 + LST_degrees) % 360, "DEC": dec})


#########################################################
def bokehTile(tileFile, jsonFile, TT=[0, 0, 0], DD=[2020, 12, 31], dynamic=False, plotTitle=''):
    citls, h = fitsio.read(tileFile, header=True)
#    w = (np.where(citls['IN_DESI'] == 1)[0])
#    inci = citls[w]
    
    if jsonFile is not None:
        with open(jsonFile, "r") as read_file:
            data = json.load(read_file)


    ## Coloring scheme
    palette = ['green', 'red', 'white']
    dye = []

    for tile in citls['TILEID']:

        rang = 2  # 'orange'
        
        if jsonFile is not None:
            if str(tile) in data:
                rang = 0  # 'green' #green default
                if len(data[str(tile)]['unassigned']) > 0:  # not assigned (red)
                    rang = 1  # 'red' #'red'
                if (0 in data[str(tile)]['gfa_stars_percam']):
                    print(data[str(tile)]['gfa_stars_percam'])
                    rang = 1  # 'cyan'
        else: rang = 0 # green if qa.json is not provided 

        dye.append(rang)

    dye = np.asarray(dye)
    w = (np.where(dye < 2)[0])
    citls = citls[w]
    dye = dye[w]
    mapper = linear_cmap(field_name='DYE', palette=palette, low=0, high=2)

    #########################################################
    TOOLS = ['pan', 'tap', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'box_select']

    obsTime = dt(DD[0], DD[1], DD[2], TT[0], TT[1], TT[2])
    # print(get_kp_twilights(TT,DD))
    
    if plotTitle=='' or plotTitle is None:
        PTITLE = ''
    else:
        PTITLE = 'Program: '+plotTitle
    
    p = figure(tools=TOOLS, toolbar_location="right", plot_width=800, plot_height=450,
               title=PTITLE, active_drag='box_select')  # str(DD[1])+" - 2019")
    p.title.text_font_size = '16pt'
    p.title.text_color = 'black'
    p.grid.grid_line_color = "gainsboro"

    ###############################  adding ecliptic plane+ hour grid ####################3
    add_plane(p, color='red', plane='ecliptic', projection='equatorial')

    
    
    tiledata = dict(
        RA = citls['RA'],
        DEC = citls['DEC'], 
        TILEID = citls['TILEID'], 
        BRIGHTRA = citls['BRIGHTRA'][:,0], 
        BRIGHTDEC = citls['BRIGHTDEC'][:,0], 
        BRIGHTVTMAG = citls['BRIGHTVTMAG'][:,0], 
        EBV_MED =  np.round(citls['EBV_MED'], 3),
        STAR_DENSITY =  citls['STAR_DENSITY'], 
        DYE =  dye,
        program = citls['PROGRAM'],
        selected = np.ones(len(citls), dtype=bool),
    )    
    

    for colname in ['STAR_DENSITY', 'EBV_MED']:
        if colname in citls.dtype.names:
            tiledata[colname] = citls[colname]
    
    tiles = ColumnDataSource(data=tiledata)
    
    
    
    colformat = bktables.NumberFormatter(format='0,0.00')
    columns = [
        bktables.TableColumn(field='TILEID', title='TILEID', width=80),
        bktables.TableColumn(field='RA', title='RA', formatter=colformat),
        bktables.TableColumn(field='DEC', title='DEC', formatter=colformat),
    ]
    
    for colname in ['STAR_DENSITY', 'EBV_MED']:
        if colname in tiledata:
            columns.append(bktables.TableColumn(field=colname, title=colname, formatter=colformat))
            
    columns.append(bktables.TableColumn(field='selected', title='Selected'))
    
    tiletable = bktables.DataTable(columns=columns, source=tiles, width=800)

    tiles.selected.js_on_change('indices', CustomJS(args=dict(s1=tiles), code="""
        var inds = cb_obj.indices;
        var d1 = s1.data;
        for (var i=0; i<d1['selected'].length; i++) {
            d1['selected'][i] = false;
        }
        for (var i = 0; i < inds.length; i++) {
            d1['selected'][inds[i]] = true;
        }
        s1.change.emit();
    """)
    )
            
    
    render = p.circle('RA', 'DEC', source=tiles, size=9, line_color='chocolate', color=mapper, alpha=0.4,
                      hover_color='orange', hover_alpha=1, hover_line_color='red',

                      # set visual properties for selected glyphs
                      selection_fill_color='orange',
                      selection_line_color='white',
                      # set visual properties for non-selected glyphs
                      nonselection_fill_alpha=0.4,
                      nonselection_fill_color=mapper)

    p.xaxis.axis_label = 'RA [deg]'
    p.yaxis.axis_label = 'Dec. [deg]'
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.grid.grid_line_color = "gainsboro"
    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"

    p.x_range = Range1d(360, 0)
    p.y_range = Range1d(-40, 95)
    p.toolbar.logo = None
    p.toolbar_location = None

    # mytext = Label(x=180, y=-35, text="S", text_color='gray', text_font_size='12pt') ; p.add_layout(mytext)
    # mytext = Label(x=180, y=88, text="N", text_color='gray', text_font_size='12pt') ; p.add_layout(mytext)
    # mytext = Label(x=350, y=45, text="E", text_color='gray', text_font_size='12pt', angle=np.pi/2) ; p.add_layout(mytext)
    # mytext = Label(x=4, y=45, text="W", text_color='gray', text_font_size='12pt', angle=np.pi/2) ; p.add_layout(mytext)

    ## Javascript code to open up custom html pages, once user click on a tile
    code = """
        var index_selected = source.selected['1d']['indices'][0];
        var tileID = source.data['TILEID'][index_selected];
        if (tileID!==undefined) {
        var win = window.open("http://www.astro.utah.edu/~u6022465/cmx/ALL_SKY/dr8/allSKY_ci_tiles/sub_pages/tile-"+tileID+".html", " ");
        try {win.focus();} catch (e){} }
    """

    taptool = p.select(type=TapTool)
    taptool.callback = CustomJS(args=dict(source=tiles), code=code)

    ## The html code for the hover window that contain tile infrormation
    ttp = """
        <div>
            <div>
                <span style="font-size: 14px; color: blue;">Tile ID:</span>
                <span style="font-size: 14px; font-weight: bold;">@TILEID{int}</span>
            </div>
            <div>
                <span style="font-size: 14px; color: blue;">RA:</span>
                <span style="font-size: 14px; font-weight: bold;">@RA</span>
            </div>  
            <div>
                <span style="font-size: 14px; color: blue;">Dec:</span>
                <span style="font-size: 14px; font-weight: bold;">@DEC</span>
            </div>     
            <div>
                <span style="font-size: 14px; color: blue;">EBV_MED:</span>
                <span style="font-size: 14px; font-weight: bold;">@EBV_MED{0.000}</span>
            </div> 
            <div>
                <span style="font-size: 14px; color: blue;">STAR_DENSITY:</span>
                <span style="font-size: 14px; font-weight: bold;">@STAR_DENSITY{0}</span>
            </div> 
            <div>
                <span style="font-size: 14px; color: blue;">BRIGHTEST_STAR_VTMAG:</span>
                <span style="font-size: 14px; font-weight: bold;">@BRIGHTVTMAG</span>
            </div> 
            <div>
                <span style="font-size: 14px; color: blue;">BRIGHTEST_STAR_LOC:</span>
                <span style="font-size: 14px; font-weight: bold;">(@BRIGHTRA, @BRIGHTDEC)</span>
            </div> 
        </div>
    """

    hover = HoverTool(tooltips=ttp, renderers=[render])

    hover.point_policy = 'snap_to_data'
    hover.line_policy = 'nearest'
    # hover.mode='vline'
    p.add_tools(hover)

    cross = CrosshairTool()
    # cross.dimensions='height'
    cross.line_alpha = 0.3
    cross.line_color = 'gray'
    p.add_tools(cross)

    # Setting the second y axis range name and range
    p.extra_y_ranges = {"foo": p.y_range}
    p.extra_x_ranges = {"joo": p.x_range}

    # Adding the second axis to the plot.
    p.add_layout(LinearAxis(y_range_name="foo"), 'right')
    p.add_layout(LinearAxis(x_range_name="joo"), 'above')

    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"

    if dynamic:

        # twilight_source = get_kp_twilights(TT,DD)   # evening and morning twilights at every TT and DD
        circleSource_1 = skyCircle(TT, DD, 1.5)
        p.circle('RA', 'DEC', source=circleSource_1, size=1.5, color='black')
        circleSource_2 = skyCircle(TT, DD, 2.0)
        p.circle('RA', 'DEC', source=circleSource_2, size=0.5, color='gray')

    else:
        circleSource = skyCircle(TT, DD, 1.5)
        p.circle('RA', 'DEC', source=circleSource, size=1.5, color=None)

    ### Dealing with the Moon and Jupiter
    inFile = 'moonLoc_jupLoc_fracPhase.csv'  # 'moon_loc_jup_loc_fracPhase_namePhase.csv'
    tbl_moon_jup = np.genfromtxt(inFile, delimiter=',', filling_values=-1, names=True, dtype=None)  # , dtype=np.float)

    loc = EarthLocation.of_site('Kitt Peak')
    kp_lat = 31, 57, 48
    kp_lon = -111, 36, 00
    mooninfo_obj = pylunar.MoonInfo((kp_lat), (kp_lon))

    m_ra, m_dec, frac_phase, name_phase = moonLoc(TT, DD, loc, mooninfo_obj)
    j_ra, j_dec = jupLoc(TT, DD, loc)

    #moonSource = ColumnDataSource({"moon_RAS": tbl_moon_jup['moon_ra'], "moon_DECS": tbl_moon_jup['moon_dec'],
    #                               "Phase_frac": tbl_moon_jup['moon_phase_frac']})
      
    moonSource = ColumnDataSource({"moon_RAS":tbl_moon_jup['moon_ra'], "moon_DECS":tbl_moon_jup['moon_dec'], "Phase_frac":np.round(100*tbl_moon_jup['moon_phase_frac'])})

    ####moon_RADEC = ColumnDataSource({"moon_ra": [m_ra.deg], "moon_dec": [m_dec.deg], "phase_frac": [frac_phase]})
    moon_RADEC_ = ColumnDataSource({"moon_ra":[m_ra.deg-360], "moon_dec":[m_dec.deg],"phase_frac":[frac_phase]})
    moon_RADEC = ColumnDataSource({"moon_ra":[m_ra.deg], "moon_dec":[m_dec.deg],"phase_frac":[frac_phase]})

    render_moon = p.circle('moon_ra', 'moon_dec', source=moon_RADEC, size=170, color='cyan', alpha=0.2)
    render_moon = p.circle('moon_ra', 'moon_dec', source=moon_RADEC, size=4, color='blue')

    render_moon = p.circle('moon_ra', 'moon_dec', source=moon_RADEC_, size=170, color='cyan', alpha=0.2)
    render_moon = p.circle('moon_ra', 'moon_dec', source=moon_RADEC_, size=4, color='blue')

    jupSource = ColumnDataSource({"jup_RAS": tbl_moon_jup['jup_ra'], "jup_DECS": tbl_moon_jup['jup_dec']})
    jup_RADEC = ColumnDataSource({"jup_ra": [j_ra.deg], "jup_dec": [j_dec.deg]})

    twilight = get_kp_twilights(TT, DD)  # evening and morning twilights at every TT and DD
    twilight_source = ColumnDataSource({"eve_twilight": [twilight[0]], "mor_twilight": [twilight[1]]})

    render_jup = p.circle('jup_ra', 'jup_dec', source=jup_RADEC, size=5, color='blue')
    render_jup = p.circle('jup_ra', 'jup_dec', source=jup_RADEC, size=4, color='gold')

    TXTsrc = ColumnDataSource(dict(x=[350], y=[85], text=['Moon Phase: ' + "%.0f" % (frac_phase * 100) + "%"]))
    glyph = Text(x="x", y="y", text="text", angle=0, text_color="black")
    p.add_glyph(TXTsrc, glyph)
    

    TXTsrc_moon = ColumnDataSource(dict(x=[m_ra.deg+10], y=[m_dec.deg-10], text=['Moon']))
    glyph = Text(x="x", y="y", text="text", angle=0, text_color="blue", text_alpha=0.3, text_font_size='10pt')
    p.add_glyph(TXTsrc_moon, glyph)

    TXTsrc_jup = ColumnDataSource(dict(x=[j_ra.deg+5], y=[j_dec.deg-8], text=['Jup.']))
    glyph = Text(x="x", y="y", text="text", angle=0, text_color="black", text_alpha=0.3, text_font_size='10pt')
    p.add_glyph(TXTsrc_jup, glyph)
     
    
    callback = CustomJS(args=dict(source_sky1=circleSource_1,source_sky2=circleSource_2, 
                                  source_moon=moonSource, source_moon_RADEC=moon_RADEC, 
                                  source_moon_RADEC_=moon_RADEC_,
                                  source_jup=jupSource, source_jup_RADEC=jup_RADEC,
                                  sourceTXT = TXTsrc,
                                  sourceTXTmoon = TXTsrc_moon,
                                  sourceTXTjup = TXTsrc_jup
                                 ), 
                        code="""
                // First set times as if they were UTC
                var t = new Date(time_slider.value);
                var d = new Date(date_slider.value);
                var data1 = source_sky1.data;
                var ra_1 = data1['RA'];
                var ra0_1 = data1['RA0'];
                
                var data2 = source_sky2.data;
                var ra_2 = data2['RA'];
                var ra0_2 = data2['RA0'];
                
                var data_moon = source_moon.data;
                var ras_moon = data_moon['moon_RAS'];
                var decs_moon = data_moon['moon_DECS'];
                var phase_frac = data_moon['Phase_frac'];
                
                var moonRADEC = source_moon_RADEC.data;
                var moon_ra = moonRADEC['moon_ra'];
                var moon_dec = moonRADEC['moon_dec'];
     
                var moonRADEC_ = source_moon_RADEC_.data;
                var moon_ra_ = moonRADEC_['moon_ra'];
                var moon_dec_ = moonRADEC_['moon_dec'];
                
                var data_jup = source_jup.data;
                var ras_jup = data_jup['jup_RAS'];
                var decs_jup = data_jup['jup_DECS'];
                
                var jupRADEC = source_jup_RADEC.data;
                var jup_ra = jupRADEC['jup_ra'];
                var jup_dec = jupRADEC['jup_dec'];


                var Hour  = t.getUTCHours();
                var Day   = d.getDate();
                var Month = d.getMonth();
                
                var Year = new Array(31,28,31,30,31,30,31,31,30,31,30,31);
                var all_FULdays = 0;
                for (var i = 0; i < Month; i++)
                    all_FULdays=all_FULdays+Year[i];
                all_FULdays = all_FULdays + (Day-1);
                
                if (Hour<12) all_FULdays=all_FULdays+1;
                
                var all_minutes = all_FULdays*24+Hour;
                
                if (all_minutes<8800) {
                    moon_ra[0] = ras_moon[all_minutes];
                    moon_dec[0] = decs_moon[all_minutes];   
                    moon_ra_[0] = ras_moon[all_minutes]-360.;
                    moon_dec_[0] = decs_moon[all_minutes];   
                                        
                }


                var jupTXTdata = sourceTXTjup.data;
                var x_jup = jupTXTdata['x'];
                var y_jup = jupTXTdata['y'];
                var text_jup = jupTXTdata['text'];  
                
                
                if (all_minutes<8800) {
                    jup_ra[0] = ras_jup[all_minutes];
                    jup_dec[0] = decs_jup[all_minutes];   
                    x_jup[0] = jup_ra[0]+5;
                    y_jup[0] = jup_dec[0]-8;                     
                }
                 
                                
                if (t.getUTCHours() < 12) {
                    d.setTime(date_slider.value + 24*3600*1000);
                } else {
                    d.setTime(date_slider.value);
                }
                d.setUTCHours(t.getUTCHours());
                d.setUTCMinutes(t.getUTCMinutes());
                d.setUTCSeconds(0);        
                
                // Correct to KPNO local time
                // d object still thinks in UTC, which is 7 hours ahead of KPNO
                d.setTime(d.getTime() + 7*3600*1000);
                // noon UT on 2000-01-01
                var reftime = new Date();
                reftime.setUTCFullYear(2000);
                reftime.setUTCMonth(0);   // Months are 0-11 (!)
                reftime.setUTCDate(1);    // Days are 1-31 (!)
                reftime.setUTCHours(12);
                reftime.setUTCMinutes(0);
                reftime.setUTCSeconds(0);
                
                // time difference in days (starting from milliseconds)
                var dt = (d.getTime() - reftime.getTime()) / (24*3600*1000);

                // Convert to LST
                var mayall_longitude_degrees = -(111 + 35/60. + 59.6/3600);
                var LST_hours = ((18.697374558 + 24.06570982441908 * dt) + mayall_longitude_degrees/15) % 24;
                var LST_degrees = LST_hours * 15;
                
                

                for (var i = 0; i < ra_1.length; i++) {
                    ra_1[i] = (ra0_1[i] + LST_degrees) % 360;
                }
                
                for (var i = 0; i < ra_2.length; i++) {
                    ra_2[i] = (ra0_2[i] + LST_degrees) % 360;
                }                

                //// Here we gtake care of the moon phasde text
                var TXTdata = sourceTXT.data;
                var x = TXTdata['x'];
                var y = TXTdata['y'];
                var text = TXTdata['text'];
                
                var moonTXTdata = sourceTXTmoon.data;
                var x_moon = moonTXTdata['x'];
                var y_moon = moonTXTdata['y'];
                var text_moon = moonTXTdata['text'];   

                // x[0] = 1;
                // y[0] = 40;
                if (all_minutes<8800) {
                    text[0] = 'Moon Phase: ' + phase_frac[all_minutes]+'%';
                    x_moon[0] = moon_ra[0]+10;
                    y_moon[0] = moon_dec[0]-10;
                }

                sourceTXT.change.emit();
                /////////////////////////////// Moon phase code ends.

                source_sky1.change.emit();
                source_sky2.change.emit();
                //source_moon_RADEC.change.emit();
                //source_moon_RADEC_.change.emit();
                //source_jup_RADEC.change.emit();
                sourceTXTmoon.change.emit();
                sourceTXTjup.change.emit();

                //alert(d);
    """)


    if dynamic:
        ### TIME
        Timeslider = DateSlider(start=dt(2019, 9, 1, 16, 0, 0), end=dt(2019, 9, 2, 8, 0, 0),
                                value=dt(2019, 9, 1, 16, 0, 0), step=1, title="KPNO local time(hh:mm)", format="%H:%M",
                                width=800)

        ## DATE
        Dateslider = DateSlider(start=dt(2019, 9, 1, 16, 0, 0), end=dt(2020, 8, 31, 8, 0, 0),
                                value=dt(2019, 10, 1, 16, 0, 0), step=1, title="Date of sunset(4pm-8am)",
                                format="%B:%d", width=800)

        callback.args['time_slider'] = Timeslider
        callback.args['date_slider'] = Dateslider

        Dateslider.js_on_change('value', callback)
        Timeslider.js_on_change('value', callback)

        layout = column(p, Dateslider, Timeslider, tiletable)
        # show(p)
        return layout

    return p


def get_fiberassign_info(fafile):
    """Extract data from a fiberassign FITS file.

    Parameters
    ----------
    fafile : str
        Fiberassign FITS file name.

    Returns
    -------
    tile_id : int
        Unique tile ID number.
    tile_ra : float
        Central tile RA, in degrees.
    tile_dec : float
        Central tile Dec, in degrees.
    fa_surv : str
        Survey type: CMX, SV0, etc.
    fa_vers : str
        Fiberassign version number.
    fa_date : str
        Fiberassign date.
    """
    head = fitsio.read_header(fafile, ext='FIBERASSIGN')
    tile_id  = head['TILEID']
    tile_ra  = head['TILERA']
    tile_dec = head['TILEDEC']
    fa_vers  = head['FA_VER']
    fa_surv  = head['FA_SURV']
    fa_date  = head['FA_DATE'] if 'FA_DATE' in head else ''

    return tile_id, tile_ra, tile_dec, fa_surv, fa_vers, fa_date


def bokeh_tiles(fiberassign_files, TT=[0,0,0], DD=[2020,12,1], dynamic=False, plot_title=''):
    """Create a bokeh plot of a list of DESI tiles.

    Parameters
    ----------
    fiberassign_files : list
        A list of fiberassign FITS files.
    TT : list of int
        Time [hr, min, sec]
    DD : list of int
        Date [yyyy, mm, dy]
    dynamic : bool
        Turn on plot dynamic elements.
    plot_title : str
        Title for celestial coordinate plot.

    Returns
    -------
    p : bokeh.plotting.figure
        Figure object for rendering in HTML + JS.
    """
    tile_id, tile_ra, tile_dec, fsurv, fvers, fdate = [],[],[],[],[],[]

    for fafile in fiberassign_files:
        _id, _ra, _dec, _surv, _vers, _date = get_fiberassign_info(fafile)
        tile_id.append(_id)
        tile_ra.append(_ra)
        tile_dec.append(_dec)
        fsurv.append(_surv.upper())
        fvers.append(_vers)
        fdate.append(_date)

    # Color map for tiles.
    dye = np.zeros_like(tile_id, dtype=int)
    palette = ['green', 'red', 'white']
    mapper = linear_cmap(field_name='DYE', palette=palette, low=0, high=2)

    # Set up plot toolbar.
    tools = ['pan', 'tap', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'box_select']
    obs_time = datetime(*DD, *TT)

    if plot_title == '' or plot_title is None:
        ptitle = ''
    else:
        ptitle = 'Program: {}'.format(plot_title)

    # Set up the main figure.
    p = figure(tools=tools, toolbar_location='right',
               plot_width=800, plot_height=450,
               title=ptitle, active_drag='box_select')
    p.title.text_font_size = '16pt'
    p.title.text_color = 'black'
    p.grid.grid_line_color = 'gainsboro'

    # Add ecliptic plane + hour grid.
    add_plane(p, color='red', plane='GeocentricTrueEcliptic', projection='ICRS')

    # Add tile table.
    tiledata = dict(
        RA = tile_ra,
        DEC = tile_dec,
        TILEID = tile_id,
        DYE = dye,
        program = fsurv,
        selected = np.ones_like(tile_ra, dtype=bool))

    tiles = ColumnDataSource(data=tiledata)

    colformat = bktables.NumberFormatter(format='0,0.00')
    columns = [
        bktables.TableColumn(field='TILEID', title='TILEID', width=80),
        bktables.TableColumn(field='RA', title='RA', formatter=colformat),
        bktables.TableColumn(field='DEC', title='DEC', formatter=colformat),
        bktables.TableColumn(field='program', title='program'),
        bktables.TableColumn(field='selected', title='selected')
    ]

    tiletable = bktables.DataTable(columns=columns, source=tiles, width=800)

    tiles.selected.js_on_change('indices', CustomJS(args=dict(s1=tiles), code="""
        var inds = cb_obj.indices;
        var d1 = s1.data;
        for (var i=0; i<d1['selected'].length; i++) {
            d1['selected'][i] = false;
        }
        for (var i = 0; i < inds.length; i++) {
            d1['selected'][inds[i]] = true;
        }
        s1.change.emit();
    """)
    )

    # Render the tiles in the plot.
    render = p.circle('RA', 'DEC', source=tiles, size=9,
                      line_color='chocolate', color=mapper, alpha=0.4,
                      hover_color='orange', hover_alpha=1, hover_line_color='red',
                      # set visual properties for selected glyphs
                      selection_fill_color='orange',
                      selection_line_color='white',
                      # set visual properties for non-selected glyphs
                      nonselection_fill_alpha=0.4,
                      nonselection_fill_color=mapper)

    p.xaxis.axis_label = 'RA [deg]'
    p.yaxis.axis_label = 'Dec. [deg]'

    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"

    p.grid.grid_line_color = "gainsboro"

    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"

    p.x_range = Range1d(360, 0)
    p.y_range = Range1d(-40, 95)

    p.toolbar.logo = None
    p.toolbar_location = None

    # JS code to open custom HTML pages when users click on a tile.
    # ...

    # HTML for hover window with tile information.
    ttip = """
        <div>
        <div>
            <span style="font-size: 14px; color: blue;">Tile ID:</span>
            <span style="font-size: 14px; font-weight: bold;">@TILEID{int}</span>
        </div>
        <div>
            <span style="font-size: 14px; color: blue;">RA:</span>
            <span style="font-size: 14px; font-weight: bold;">@RA</span>
        </div>
        <div>
            <span style="font-size: 14px; color: blue;">Dec:</span>
            <span style="font-size: 14px; font-weight: bold;">@DEC</span>
        </div>
        <div>
            <span style="font-size: 14px; color: blue;">Prog:</span>
            <span style="font-size: 14px; font-weight: bold;">@program</span>
        </div>
    </div>
    """

    hover = HoverTool(tooltips=ttip, renderers=[render])
    hover.point_policy = 'snap_to_data'
    hover.line_policy = 'nearest'
    p.add_tools(hover)

    # Add a crosshairs to the plot.
    cross = CrosshairTool()
    cross.line_alpha = 0.3
    cross.line_color = 'gray'
    p.add_tools(cross)

    # Set the mirrored y axis name and range and add to the plot.
    p.extra_y_ranges = {'foo' : p.y_range}
    p.extra_x_ranges = {'joo' : p.x_range}

    p.add_layout(LinearAxis(y_range_name='foo'), 'right')
    p.add_layout(LinearAxis(x_range_name='joo'), 'above')

    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    # Add dynamically moving observatory FOV.
    if dynamic:
        ra, dec = observer.airmass_limit_range(1.5, Time(obs_time))
        circ1 = ColumnDataSource({'RA': ra, 'DEC': dec})
        p.circle('RA', 'DEC', source=circ1, size=1.5, color='black')

        ra, dec = observer.airmass_limit_range(2.0, Time(obs_time))
        circ2 = ColumnDataSource({'RA': ra, 'DEC': dec})
        p.circle('RA', 'DEC', source=circ2, size=0.5, color='black')
    else:
        ra, dec = observer.airmass_limit_range(1.5, Time(obs_time))
        circ1 = ColumnDataSource({'RA': ra, 'DEC': dec})
        p.circle('RA', 'DEC', source=circ1, size=1.5, color='black')

    # Moon position.
    m_ra, m_dec, frac_phase = observer.get_moon(Time(obs_time))
    print(obs_time)
    print(m_ra, m_dec)

    moon_RADEC_ = ColumnDataSource({"moon_ra":[m_ra-360], "moon_dec":[m_dec],"phase_frac":[frac_phase]})
    moon_RADEC = ColumnDataSource({"moon_ra":[m_ra], "moon_dec":[m_dec],"phase_frac":[frac_phase]})

    # Draw moon with excursion circle.
    render_moon = p.circle('moon_ra', 'moon_dec', source=moon_RADEC, size=170, color='cyan', alpha=0.2)
    render_moon = p.circle('moon_ra', 'moon_dec', source=moon_RADEC, size=4, color='blue')

    # Draw text with moon phase and label the moon.
    txt_phase = ColumnDataSource(dict(x=[350], y=[85],
        text=['Moon Phase: ' + '%.0f' % (frac_phase * 100) + '%']))
    glyph = Text(x='x', y='y', text='text', angle=0, text_color='black')
    p.add_glyph(txt_phase, glyph)

    txt_moon = ColumnDataSource(dict(x=[m_ra+10], y=[m_dec-10],
        text=['Moon']))
    glyph = Text(x='x', y='y', text='text', angle=0, text_color='blue',
        text_alpha=0.3, text_font_size='10pt')
    p.add_glyph(txt_moon, glyph)

    # Custom callbacks for date/time sliders.
    callback = CustomJS(args=dict(source_sky1=circ1, source_sky2=circ2, 
#                                  source_moon=moonSource,
                                  source_moon_RADEC=moon_RADEC, 
                                  source_moon_RADEC_=moon_RADEC_,
#                                  source_jup=jupSource, source_jup_RADEC=jup_RADEC,
                                  sourceTXT = txt_phase,
                                  sourceTXTmoon = txt_moon,
#                                  sourceTXTjup = TXTsrc_jup
                                 ), 
                        code="""
                // First set times as if they were UTC
                var t = new Date(time_slider.value);
                var d = new Date(date_slider.value);
                var data1 = source_sky1.data;
                var ra_1 = data1['RA'];
                var ra0_1 = data1['RA0'];
                
                var data2 = source_sky2.data;
                var ra_2 = data2['RA'];
                var ra0_2 = data2['RA0'];
                
//                var data_moon = source_moon.data;
//                var ras_moon = data_moon['moon_RAS'];
//                var decs_moon = data_moon['moon_DECS'];
//                var phase_frac = data_moon['Phase_frac'];
                
                var moonRADEC = source_moon_RADEC.data;
                var moon_ra = moonRADEC['moon_ra'];
                var moon_dec = moonRADEC['moon_dec'];
     
                var moonRADEC_ = source_moon_RADEC_.data;
                var moon_ra_ = moonRADEC_['moon_ra'];
                var moon_dec_ = moonRADEC_['moon_dec'];
                
//                var data_jup = source_jup.data;
//                var ras_jup = data_jup['jup_RAS'];
//                var decs_jup = data_jup['jup_DECS'];
                
//                var jupRADEC = source_jup_RADEC.data;
//                var jup_ra = jupRADEC['jup_ra'];
//                var jup_dec = jupRADEC['jup_dec'];

                var Hour  = t.getUTCHours();
                var Day   = d.getDate();
                var Month = d.getMonth();
                
                var Year = new Array(31,28,31,30,31,30,31,31,30,31,30,31);
                var all_FULdays = 0;
                for (var i = 0; i < Month; i++)
                    all_FULdays=all_FULdays+Year[i];
                all_FULdays = all_FULdays + (Day-1);
                
                if (Hour<12) all_FULdays=all_FULdays+1;
                
                var all_minutes = all_FULdays*24+Hour;
                
                if (all_minutes<8800) {
                    moon_ra[0] = ras_moon[all_minutes];
                    moon_dec[0] = decs_moon[all_minutes];   
                    moon_ra_[0] = ras_moon[all_minutes]-360.;
                    moon_dec_[0] = decs_moon[all_minutes];   
                }

//                var jupTXTdata = sourceTXTjup.data;
//                var x_jup = jupTXTdata['x'];
//                var y_jup = jupTXTdata['y'];
//                var text_jup = jupTXTdata['text'];  
                
//                if (all_minutes<8800) {
//                    jup_ra[0] = ras_jup[all_minutes];
//                    jup_dec[0] = decs_jup[all_minutes];   
//                    x_jup[0] = jup_ra[0]+5;
//                    y_jup[0] = jup_dec[0]-8;                     
//                }

                if (t.getUTCHours() < 12) {
                    d.setTime(date_slider.value + 24*3600*1000);
                } else {
                    d.setTime(date_slider.value);
                }
                d.setUTCHours(t.getUTCHours());
                d.setUTCMinutes(t.getUTCMinutes());
                d.setUTCSeconds(0);        
                
                // Correct to KPNO local time
                // d object still thinks in UTC, which is 7 hours ahead of KPNO
                d.setTime(d.getTime() + 7*3600*1000);
                // noon UT on 2000-01-01
                var reftime = new Date();
                reftime.setUTCFullYear(2000);
                reftime.setUTCMonth(0);   // Months are 0-11 (!)
                reftime.setUTCDate(1);    // Days are 1-31 (!)
                reftime.setUTCHours(12);
                reftime.setUTCMinutes(0);
                reftime.setUTCSeconds(0);
                
                // time difference in days (starting from milliseconds)
                var dt = (d.getTime() - reftime.getTime()) / (24*3600*1000);

                // Convert to LST
                var mayall_longitude_degrees = -(111 + 35/60. + 59.6/3600);
                var LST_hours = ((18.697374558 + 24.06570982441908 * dt) + mayall_longitude_degrees/15) % 24;
                var LST_degrees = LST_hours * 15;

                for (var i = 0; i < ra_1.length; i++) {
                    ra_1[i] = (ra0_1[i] + LST_degrees) % 360;
                }

                for (var i = 0; i < ra_2.length; i++) {
                    ra_2[i] = (ra0_2[i] + LST_degrees) % 360;
                }                

                //// Here we gtake care of the moon phasde text
                var TXTdata = sourceTXT.data;
                var x = TXTdata['x'];
                var y = TXTdata['y'];
                var text = TXTdata['text'];
                
                var moonTXTdata = sourceTXTmoon.data;
                var x_moon = moonTXTdata['x'];
                var y_moon = moonTXTdata['y'];
                var text_moon = moonTXTdata['text'];   

                // x[0] = 1;
                // y[0] = 40;
                if (all_minutes<8800) {
                    text[0] = 'Moon Phase: ' + phase_frac[all_minutes]+'%';
                    x_moon[0] = moon_ra[0]+10;
                    y_moon[0] = moon_dec[0]-10;
                }

                sourceTXT.change.emit();
                /////////////////////////////// Moon phase code ends.

                source_sky1.change.emit();
                source_sky2.change.emit();
                //source_moon_RADEC.change.emit();
                //source_moon_RADEC_.change.emit();
                //source_jup_RADEC.change.emit();
                sourceTXTmoon.change.emit();
//                sourceTXTjup.change.emit();

                //alert(d);
    """)

    # Set up date and time slider.
    if dynamic:
        time_slider = DateSlider(start=datetime(2020,12,3,16,0,0),
                                end=datetime(2020,12,4,8,0,0),
                                value=datetime(2020,12,3,16,0,0), step=1,
                                title='KPNO local time(hh:mm)', format='%H:%M',
                                width=800)

        ## DATE
        date_slider = DateSlider(start=datetime(2020, 12, 1, 16, 0, 0),
                                 end=datetime(2021, 8, 31, 8, 0, 0),
                                 value=datetime(2020, 12, 3, 16, 0, 0), step=1,
                                 title='Date of sunset(4pm-8am)', format='%B:%d',
                                 width=800)

        callback.args['time_slider'] = time_slider
        callback.args['date_slider'] = date_slider

        date_slider.js_on_change('value', callback)
        time_slider.js_on_change('value', callback)

        layout = column(p, date_slider, time_slider, tiletable)
#        layout = column(p, tiletable)
        show(layout)
        return layout

    return p


if __name__ == "__main__":
    # Set up command line interface.
    p = ArgumentParser(description='DESI Tile Picker Visualization Tool',
                       formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('input', nargs='+',
                   help='fiberassign FITS file(s)')
    p.add_argument('-j', '--json', dest='json',
                   help='qa json file (optional)')
    p.add_argument('-o', '--output', dest='output',
                   help='output html file (optional)')
    p.add_argument('-t', '--title', dest='page_title',
                   default='DESI Tile Picker',
                   help='HTML title (optional)')
    p.add_argument('-p', '--ptitle', dest='plot_title',
                   default=None,
                   help='plot title (optional)')
    p.add_argument('-x', '--xfile', dest='textfile',
                   help='Text file to be printed on the right side of plot')

    args = p.parse_args()

    # Spin through fiberassign input files (>= 1 file needed as input).
    p = bokeh_tiles(args.input, TT=[12,0,0], DD=[2020,12,3], dynamic=True, plot_title=args.plot_title)

#    script, div = components(p)
#    script = '\n'.join(['' + line for line in script.split('\n')])
#
#    if args.output is None:
#        html_name = 'tilepicker.html'
#    else:
#        html_name = args.output
#    print('The output HTML file is {}'.format(html_name))
#
#    head = html_header(args.page_title)
#    tail = html_footer()
#
#    with open(html_name, "w") as text_file:
#        text_file.write(head)
#        text_file.write('<table><tr>')
#        
#        text_file.write('<td valign="top" width="850">')
#        text_file.write(script)
#        text_file.write(div)
#        text_file.write('</td>')
#
#        if args.textfile is not None:
#            textFile = args.textfile
#            if os.path.exists(textFile):
#                f = open(textFile, 'r')
#                txt = f.read()
#                text_file.write('<td valign="top" width="400"><p>'+txt+'</p></td>')
#            else:
#                print('Warning: '+textFile+' does NOT exist. Continuing without this file ...')
#                print('         '+'You can try again using the correct text file.\n')
#
#        text_file.write('</tr></table>')
#
#        text_file.write(tail)
