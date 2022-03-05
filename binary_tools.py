import numpy as np

# plot modules
import matplotlib.pyplot as plt

# Ephemerides libraries
from astroplan import Observer, FixedTarget
import datetime
from astropy.time import Time
import sympy as sp

plt.ion()
from datetime import datetime as dt
import time
import julian

def toYearFraction(date):
    '''
    Conversion of the date into decimal year. 
    ; param date : Date, generated from datetime. Ex : datetime.today() // datetime(2021,5,25,3,0,0)
    ; type date : datetime
    '''

    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def get_several_parallactic_angles(starname, date_start, date_end, n_elements,
                                   display=False):
    date_start = Time(date_start, format='isot', scale='local')
    date_end = Time(date_end, format='isot', scale='local')

    dt = date_end - date_start
    times = date_start + dt * np.linspace(0, 1, n_elements+1, True)

    pa, pad = [], []

    for t in times:
        para_angle, chuck_angle = \
            get_parallactic_angle(starname, t.datetime.year, t.datetime.month,
                                  t.datetime.day, t.datetime.hour,
                                  t.datetime.minute, t.datetime.second,
                                  display=display)[:-1]
        pa.append(para_angle)
        pad.append(chuck_angle)

    return pa, pad

def get_parallactic_angle(starname, year, month, day, hours, minutes, seconds,
                          display=False):
    """
    Get the parallactic angle calculated by Astroplan, using celestial\
    coordinates and angle hour.

    The formula is taken from https://en.wikipedia.org/wiki/Parallactic_angle

    :param starname: Name of the star (Common name or from any catalog known
                                       by SIMBAD)
    :type starname: string
    :param year: year of the observation, in HST time zone
    :type year: int
    :param month:  month of the observation, in HST time zone
    :type month: int
    :param day:  day of the observation, in HST time zone
    :type day: int
    :param hours:  hour of the observation, in HST time zone
    :type hours: int
    :param minutes:  minute of the observation, in HST time zone
    :type minutes: int
    :param seconds:  seconds of the observation, in HST time zone
    :type seconds: int
    :param display: If True, displays the parallactic angle and the PAD
    (Parallactic angle in Chuck), defaults to False
    :type display: bool, optional
    :return: Parallactic angle in degree and the UTC time of the observation
    :rtype: tuple

    """
    # Theoretical Parallactic angle
    # Select site of observation
    subaru = Observer.at_site("Subaru", timezone="US/Hawaii")

    # Select date
    date_time = datetime.datetime(year, month, day, hours, minutes, seconds)
    utc_time = subaru.datetime_to_astropy_time(date_time)  # convert to UTC

    # Select Target
    star = FixedTarget.from_name(starname)

    # Compute parallactic angle = angle between Altaz orientation and North
    para_angle = subaru.parallactic_angle(utc_time, star).deg
    if display is True:
        print('Parallactic angle :'+str(para_angle)+' degrees')

    # Chuck's PAD
    chuck_pad = para_angle + 180 - 39
    if display is True:
        print('Parallactic angle for Chuck (PAD) :'+str(chuck_pad)+' degrees')

    return para_angle, chuck_pad, utc_time


def get_parallactic_angle_subaru(Az, El, lat=None, display=False):
    """
    Get the parallactic angle as coded in AO-188 and from the telemetry
    saved by it.
    Note, the parallactic given here is different from the one given by
    ``get_parallactic_angle''.
    It is not confirmed yet but here are two possible reasons:
        * the numerical errors (Astroplan formula uses lots of trigonometry)
        * the levation calculated in AO-188 is not the exact same one given by
        Astroplan for the exact same time of observation

    :param Az: Azimuth of the object
    :type Az: float
    :param El: Elevation of the object
    :type El: float
    :param lat: latitude of the observing site. If None, the hard coded value
    in AO-188 (Subaru) is used, defaults to None
    :type lat: float, optional
    :param display: If True, displays the parallactic angle and the PAD
    (Parallactic angle in Chuck), defaults to False
    :type display: bool, optional
    :return: Chuck PAD in degrees
    (Parallactic angle in Chuck coordinates base),
    parallactic angle as calculated in AO-188 in degree.
    :rtype: tuple

    """
    # Formula from Kudo-san to compute the PAD on chuck
    # from telescope AzEl info - from AO188 telemetry
    if lat is None:
        lat = 19.823806 * np.pi / 180
    else:
        lat = lat * np.pi / 180
    Az = (Az) * np.pi / 180
    El = El * np.pi / 180

    deg = np.sin(El) * np.cos(Az) + np.cos(El)*np.sin(lat)/np.cos(lat)
    para_angle_Subaru = np.arctan2(np.sin(Az), deg) * 180 / np.pi

    # Chuck's PAD
    chuck_pad = para_angle_Subaru + 180 - 39
    if display is True:
        print('Subaru parallactic angle for Chuck (PAD) :' +
              str(chuck_pad)+' degrees')

    return chuck_pad, para_angle_Subaru

def binary_orbit_calc(t, smaj, ecc, inc, small_om, big_om):
    """
    Calculate the orbit of a companion around the host in a binary system.

    :param smaj: length of the semi-major axis, in arcsecond
    :type smaj: float
    :param ecc: inclination of the orbit
    :type ecc: float
    :param inc: inclination of the orbit
    :type inc: float
    :param small_om: argument of the periastron, in degree
    :type small_om: float
    :param big_om: Position angle of the ascending node, in degree.
    :type big_om: float
    :param n_points: number of points to calculate the orbit.
    :type n_points: int
    :param display: plot the orbit, defaults to False
    :type display: bool, optional
    :return: x and y coordinates of the orbite
    :rtype: tuple

    Credits:
        * Author : Guillaume Duchene. Translated into Python by S. Vievard
        * Documentation by M-.A. Martinod
    """

    # Conversions
    inc = np.deg2rad(inc)
    small_om = np.deg2rad(small_om)
    big_om = np.deg2rad(big_om)

    # Defining intermediate variables
    exq = np.sqrt(1.-ecc*ecc)
    uu1 = np.cos(small_om)*np.cos(big_om) - \
        np.sin(small_om)*np.sin(big_om)*np.cos(inc)
    uu2 = -1.*np.sin(small_om)*np.cos(big_om) - \
        np.cos(small_om)*np.sin(big_om)*np.cos(inc)
    uu3 = np.cos(small_om)*np.sin(big_om) + \
        np.sin(small_om)*np.cos(big_om)*np.cos(inc)
    uu4 = -1.*np.sin(small_om)*np.sin(big_om) + \
        np.cos(small_om)*np.cos(big_om)*np.cos(inc)

    # Computing positions along the orbit
    u = 4.*np.pi*t
    x_pos = (uu1*(np.cos(u)-ecc)*smaj+uu2*smaj*exq*np.sin(u))
    y_pos = (uu3*(np.cos(u)-ecc)*smaj+uu4*smaj*exq*np.sin(u))

    return x_pos, y_pos


def binary_orbit(n_points, t0, T, smaj, ecc, inc, small_om, big_om,
                 display=False):
    """
    Plot the orbit of a binary system

    :param n_points: Number of points to plot
    :type n_points: int
    :param smaj: semi-major axis, in arcsecond
    :type smaj: float
    :param ecc: eccentricity
    :type ecc: float
    :param inc: inclination
    :type inc: float
    :param small_om: argument of the periastron, in degree
    :type small_om: float
    :param big_om: Position angle of the ascending node, in degree.
    :type big_om: float
    :param display: plot the orbit, defaults to False
    :type display: bool, optional
    :return: epoch, x and y coordinates of the ``n_points''
    positions on the orbit
    :rtype: tuple

    """

    x_pos = np.zeros(n_points)
    y_pos = np.zeros(n_points)
    epoch = np.zeros(n_points)

    nn = 2.*np.pi/T

    for i in range(n_points):
        t = i/(n_points-1)
        u = 4.*np.pi*t
        epoch[i] = t0+(u-ecc*np.sin(u))/nn
        x_pos[i], y_pos[i] = binary_orbit_calc(
            t, smaj, ecc, inc, small_om, big_om)

    # Plot orbit
    if display is True:
        plt.figure()
        plt.plot(y_pos, x_pos)
        plt.ylabel('Declination (arcsec)')
        plt.xlabel("Right ascension (arcsec)")
        plt.xlim([0.06, -0.06])
        plt.ylim([-0.06, 0.06])
        plt.scatter(0, 0, c='black', marker='+')
        plt.axis('equal')

    return epoch, x_pos, y_pos


def binary_ephem(current_date, t0, T, smaj, ecc, inc, small_om, big_om, t0_err,
                 T_err, smaj_err, ecc_err, inc_err, small_om_err, big_om_err,
                 nsamp, display=False):
    """
    Gives the separation and PA of the companion at the date of observation.

    :param current_date: Date of observation, in decimal year
    :type current_date: float
    :param t0: Periastron date, in decimal year
    :type t0: float
    :param T: Period, in decimal year
    :type T: float
    :param smaj: semi-major axis, in arcsecond
    :type smaj: float
    :param ecc: eccentricity
    :type ecc: float
    :param inc: inclination
    :type inc: float
    :param small_om: argument of the periastron, in degree
    :type small_om: float
    :param big_om: Position angle of the ascending node, in degree.
    :type big_om: float
    :param display: plot the orbit, defaults to False
    :type display: bool, optional
    :return: separation in arcsecond and the position angle in degree
    :rtype: tuple

    """

    sep, pa = [], []
    if nsamp <= 0:
        nsamp = 1

    for k in range(nsamp):
        if nsamp <= 1 or k == 0:
            t0_samp = t0
            smaj_samp = smaj
            ecc_samp = ecc
            inc_samp = inc
            small_om_samp = small_om
            big_om_samp = big_om
            T_samp = T
        else:
            t0_samp = np.random.normal(t0, t0_err)
            smaj_samp = np.random.normal(smaj, smaj_err)
            ecc_samp = np.random.normal(ecc, ecc_err)
            inc_samp = np.random.normal(inc, inc_err)
            small_om_samp = np.random.normal(small_om, small_om_err)
            big_om_samp = np.random.normal(big_om, big_om_err)
            T_samp = np.random.normal(T, T_err)
        # Time spent after passing periastron
        date_sub = np.mod((current_date-t0_samp), T_samp)
        # Solve equation to find the actual time locating
        # the position of the companion
        x = sp.symbols('x', real=True)
        eq = sp.Eq(4*sp.pi*x-ecc_samp*sp.sin(4*sp.pi*x),
                   2*sp.pi/T_samp*date_sub)
        t = sp.nsolve(eq, x, 0)
        t = float(t)

        # Load orbit info
        x_date, y_date = binary_orbit_calc(t, smaj_samp, ecc_samp, inc_samp,
                                           small_om_samp, big_om_samp)

        # Compute Sep. and Position Angle (PA)
        sep.append(np.sqrt(x_date**2+y_date**2))
        pa.append(np.rad2deg(np.arctan2(y_date, x_date)) % 360)

    sep = np.array(sep) * 1000
    pa = np.array(pa)

    print('Binary separation: %.5f +/- %.5f mas' % (sep[0], sep.std()))
    print('Binary Position Angle: %.5f +/- %.5f degrees' %
          (pa[0], pa.std()))

    if display is True:
        x_pos, y_pos = binary_orbit(
            1000, t0, T, smaj, ecc, inc, small_om, big_om, display=False)[1:]

        plt.figure()
        plt.plot(y_pos, x_pos)
        plt.plot(y_date, x_date, c='red', marker='+')
        plt.ylabel('Declination (arcsec)')
        plt.xlabel("Right ascension (arcsec)")
        plt.xlim([0.06, -0.06])
        plt.ylim([-0.06, 0.06])
        plt.scatter(0, 0, c='black', marker='+')
        plt.axis('equal')

    return sep, pa


def get_Orbit_Param(target):
    """
    Provides orbit parameters for binary systems. Info can be found on the website : http://www.astro.gsu.edu/wds/orb6/orb6orbits.html
    Info per column :
    HD/HIP name  ||   Vmag  ||  Period  ||  Semi-major axis  ||  Inclination  ||  Position angle of ascending node (Big Om)  ||  Time of periastron  ||  Excentricity  ||  Argument of periastron

    """
    target = target.lower()

    if target == 'capella':
        # Source: https://ui.adsabs.harvard.edu/abs/2015ApJ...807...26T/abstract
        # Time of Periastron [years]
        t0 = 1990.6989041095892
        t0_err = 0.007123287671674916
        # Semi-major axis [asec]
        smaj = 0.056442
        smaj_err = 0.023e-3
        # Eccentricity
        ecc = 0.00089
        ecc_err = 0.00011
        # Inclination [degrees]
        inc = 137.156
        inc_err = 0.046
        # Argument of periastron
        small_om = 342.6
        small_om_err = 9.
        # Position angle of ascending node
        big_om = 40.522
        big_om_err = 0.039
        # Period [years]
        T = 0.28479474332648874
        T_err = 4.3805612594113626e-07

    elif target == 'AlfEqu':
        # Time of Periastron [years]
        t0         = toYearFraction(julian.from_jd(47592.1,"mjd"))
        # Semi-major axis [asec]
        smaj       = 0.011987
        # Eccentricity
        ecc        = 0.0056
        # Inclination [degrees]
        inc        = 151.5
        # Argument of periastron
        small_om   = 342.6
        # Position angle of ascending node
        big_om     = 33.9
        # Period [years]
        T          = 98.800/365.2425  

    elif target == '47oph':
        # Time of Periastron [years]
        t0         = 1990.5795
        # Semi-major axis [asec]
        smaj       = 0.00799
        # Eccentricity
        ecc        = 0.481
        # Inclination [degrees]
        inc        = 59.5
        # Argument of periastron
        small_om   = 270.
        # Position angle of ascending node
        big_om     = 121.8
        # Period [years]
        T          = 0.071972603


    elif target == 'alpCrb':
        # Time of Periastron [years]
        t0         = 1958.497657768516
        t0_err = 0.
        # Semi-major axis [asec]
        # smaj       = 0.00175
        smaj       = 0.00866
        smaj_err = 0.
        # Eccentricity
        ecc        = 0.37
        ecc_err = 0.
        # Inclination [degrees]
        inc        = 88.2
        inc_err = 0.
        # Argument of periastron
        small_om   = 311
        small_om_err = 0.
        # Position angle of ascending node
        big_om     = 330.4
        big_om_err = 0.
        # Period [years]
        T          = 0.047528815879534565
        T_err = 0.

    elif  target == 'betaHer':
        # Time of Periastron [years]
        t0 = Time('2415500.4', format='jd')
        t0 = t0.decimalyear
        t0_err = 0.
        # Semi-major axis [asec]
        smaj = 0.01137
        smaj_err = 0.
        # Eccentricity
        ecc = 0.55
        ecc_err = 0.
        # Inclination [degrees]
        inc = 53.8
        inc_err = 0.
        # Argument of periastron
        small_om = 24.6
        small_om_err = 0.
        # Position angle of ascending node
        big_om = 341.9
        big_om_err = 0.
        # Period [years]
        T = 410.6/365.25
        T_err = 0.

    elif target == 'delsge':
        # Time of Periastron [years]
        t0 = 1979.93
        t0_err = 0.
        # Semi-major axis [asec]
        smaj = 0.051
        smaj_err = 0.
        # Eccentricity
        ecc = 0.44
        ecc_err = 0.
        # Inclination [degrees]
        inc = 140
        inc_err = 0.
        # Argument of periastron
        small_om = 257.7
        small_om_err = 0.
        # Position angle of ascending node
        big_om = 170.2
        big_om_err = 0.
        # Period [years]
        T = 10.11
        T_err = 0.

    elif target == 'HD44927':
        # Time of Periastron [years]
        t0         = 2020.594
        # Semi-major axis [asec]
        smaj       = 0.087
        # Eccentricity
        ecc        = 0.074
        # Inclination [degrees]
        inc        = 82.5
        # Argument of periastron
        small_om   = 236.8
        # Position angle of ascending node
        big_om     = 149.3
        # Period [years]
        T          = 42.841

    else:
        print('Target not listed...')

    return t0, smaj, ecc, inc, small_om, big_om, T,\
        t0_err, smaj_err, ecc_err, inc_err, small_om_err, big_om_err, T_err


def xy2spa(x,dx,y,dy, print=False):
    """
    Converts Cartesian object coordinates into polar coordinates
    (with error bars)

    :param x: x position, in arcsecond
    :type x: float
    :param dx: error bar on the x position, in arcsecond
    :type dx: float
    :param y: y position, in arcsecond
    :type y: float
    :param dy: error bar on the x position, in arcsecond
    :type dy: float
    :param print: prints the computed separation and position angle, defaults to False
    :type print: bool, optional
    :return: separation and error bar in arcsecond and the position angle and error bar in degree
    :rtype: tuple

    """
	# Compute the separation from the (x,y) coordinates
    sep = np.around(np.sqrt(x**2+y**2),decimals=1)
    # Compute the error on the separation
    dsep = abs(np.around(sep * 0.5*(dx/x) + sep* 0.5*(dy/y),decimals=1))
    # Compute the angle from the (x,y) coordinates
    angle = np.around(np.degrees(np.arctan2(y,x)),decimals=1)
    # Compute the error on the angle
    dangle = np.around(np.degrees(abs((1/x)/(1+y**2))*dy + abs( (-y/(x**2))*1/(1+(y/x)**2) )*dx),decimals=1)

    if print is True : 
        print('Separation : '+str(sep)+' +/- '+str(dsep))
        print('Position Angle : '+str(angle)+' +/- '+str(dangle))

    return sep, dsep, angle, dangle 


def spa2xy(r,dr,t,dt, display = False):
    """
    Converts polar object coordinates into Cartesian coordinates
    (with error bars)

    :param r: separation, in arcsecond
    :type r: float
    :param dr: error bar on the separation, in arcsecond
    :type dr: float
    :param t: position angle, in degree
    :type t: float
    :param dt: error bar on the position angle, in degree
    :type dt: float
    :param print: prints the computed Cartesian coordinates, defaults to False
    :type print: bool, optional
    :return: Cartesian coordinates and error bars in arcsecond 
    :rtype: tuple

    """
    # Compute the x cordinate from the (r,t) coordinates
    x = r*np.cos(np.deg2rad(t))
    # Compute the error on x
    dx = np.cos(np.deg2rad(t))*dr - r*np.sin(np.deg2rad(t))*np.deg2rad(dt)
    # Compute the y cordinate from the (r,t) coordinates
    y = r*np.sin(np.deg2rad(t))
    # Compute the error on y
    dy = np.sin(np.deg2rad(t))*dr + r*np.cos(np.deg2rad(t))*np.deg2rad(dt)

    if display is True :
        print('X position : '+str(x)+' +/- '+str(dx))
        print('Y position : '+str(y)+' +/- '+str(dy))

    return x, dx, y, dy 









if __name__ == "__main__":
    # =============================================================================
    # Capella
    # =============================================================================
    # # Load Orbits params
    # t0, smaj, ecc, inc, small_om, big_om, T,\
    #     t0_err, smaj_err, ecc_err, inc_err, small_om_err, big_om_err, T_err =\
    #     get_Orbit_Param('Capella')

    # # Orbit
    # # epoch, x_pos, y_pos = binary_orbit(1000, t0, T, smaj, ecc, inc, small_om,
    # #                                    big_om, display = False)

    # date = Time('2020-09-17T04:50:18', format='isot')
    # subaru = Observer.at_site("Subaru", timezone="US/Hawaii")

    # # Select date
    # utc_time = subaru.datetime_to_astropy_time(date.datetime)  # convert to UTC
    # # utc_time = Time(datetime.datetime.now())
    # current_date = utc_time.decimalyear

    # # Compute / display binary ephem
    # sep, pa = binary_ephem(current_date, t0, T, smaj, ecc, inc, small_om,
    #                        big_om, t0_err, T_err, smaj_err, ecc_err, inc_err,
    #                        small_om_err, big_om_err, nsamp=1000, display=True)

    # # Get the Parallactic Angle from astroplan library
    # PA = get_parallactic_angle('Capella', 2020, 9, 17, 4, 50, 18, display=True)
    # Az = 17.679
    # El = 62.0691
    # Az = 2.0234e+02  # For 2020-09-17:4h50h18s HST (+10h for UTC)
    # El = 6.0896e+01  # For 2020-09-17:4h50h18s HST (+10h for UTC)

    # PAD, subaru_q = get_parallactic_angle_subaru(Az, El, display=True)

    # =============================================================================
    # Del sge
    # =============================================================================
    # Load Orbits params
    t0, smaj, ecc, inc, small_om, big_om, T,\
        t0_err, smaj_err, ecc_err, inc_err, small_om_err, big_om_err, T_err =\
        get_Orbit_Param('delsge')
    date = Time('2021-04-30T05:00:00', format='isot')
    subaru = Observer.at_site("Subaru", timezone="US/Hawaii")

    # Select date
    utc_time = subaru.datetime_to_astropy_time(date.datetime)  # convert to UTC
    # utc_time = Time(datetime.datetime.now())
    current_date = utc_time.decimalyear

    # Compute / display binary ephem
    sep, pa = binary_ephem(current_date, t0, T, smaj, ecc, inc, small_om,
                           big_om, t0_err, T_err, smaj_err, ecc_err, inc_err,
                           small_om_err, big_om_err, nsamp=1000, display=True)

    # Get the Parallactic Angle from astroplan library
    PA = get_parallactic_angle('del Sge', 2021, 4, 30, 5, 00, 00, display=True)

    pa_list, pad_list = get_several_parallactic_angles(
        'del Sge', '2021-04-30T04:32:04.0', '2021-04-30T05:38:33.0', 13)

    # Get the Parallactic Angle from Telescope position
    '''
    WARNING : IF Az comes from AO188 telemetry,
    need to subtract 180 degrees before using get_parallactic_angle_subaru()
    '''
    # Az = 17.679
    # El = 62.0691

    # PAD, subaru_q = get_parallactic_angle_subaru(Az, El, display = True)

    # date = Time('2020-09-16T14:48:32', format='isot')
    # print('Seb', get_parallactic_angle('Capella', date.datetime.year, date.datetime.month, date.datetime.day, \
    #                               date.datetime.hour-10, date.datetime.minute, date.datetime.second)[0])

    # subaru = Observer.at_site("Subaru", timezone="US/Hawaii")
    # star=FixedTarget.from_name("Capella")
    # print(subaru.altaz(date, star).alt.deg, subaru.altaz(date, star).az.deg)
    # print('Astroplan', subaru.parallactic_angle(date, star).deg)
    # print('Subaru', get_parallactic_angle_subaru(subaru.altaz(date, star).az.deg, subaru.altaz(date, star).alt.deg, display = False, lat=subaru.location.lat.deg)[1])

    # LST = date.sidereal_time('mean', longitude=subaru.location.lon)
    # H = (LST - star.ra).radian
    # latitude = subaru.location.lat.radian
    # declinaison = star.dec.radian
    # sin_parallactic_angle = np.sin(H) * np.cos(latitude)
    # cos_parallactic_angle = np.cos(declinaison) * np.sin(latitude) - np.sin(declinaison) * np.cos(latitude) * np.cos(H)
    # parallactic_angle = np.arctan2(sin_parallactic_angle, cos_parallactic_angle) * 180/np.pi
    # print('A la main', parallactic_angle)
