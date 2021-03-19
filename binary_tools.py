import numpy as np

### plot modules
import matplotlib.pyplot as plt

### Ephemerides libraries
from astroplan import Observer, FixedTarget
import datetime
from astropy.time import Time
import sympy as sp

plt.ion()


def get_Parallactic_Angle(starname, year,month,day,hours,minutes,seconds, display = False):
    """
    Get the parallactic angle calculated by Astroplan, using celestial coordinates and angle hour.
    The formula is taken from https://en.wikipedia.org/wiki/Parallactic_angle
    
    :param starname: Name of the star (Common name or from any catalog known by SIMBAD)
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
    :param display: If True, displays the parallactic angle and the PAD (Parallactic angle in Chuck), defaults to False
    :type display: bool, optional 
    :return: Parallactic angle in degree and the UTC time of the observation
    :rtype: tuple

    """
    ## Theoretical Parallactic angle
    ## Select site of observation
    subaru = Observer.at_site("Subaru", timezone="US/Hawaii")

    ## Select date
    date_time = datetime.datetime(year,month,day,hours,minutes,seconds)
    utc_time = subaru.datetime_to_astropy_time(date_time) # convert to UTC

    ## Select Target
    star=FixedTarget.from_name(starname)

    ## Compute parallactic angle = angle between Altaz orientation and North
    para_angle = subaru.parallactic_angle(utc_time, star).deg
    if display is True:
        print('Parallactic angle :'+str(para_angle)+' degrees')

    ## Chuck's PAD 
    chuck_pad = para_angle +180 - 39
    if display is True:
        print('Parallactic angle for Chuck (PAD) :'+str(chuck_pad)+' degrees')

    return para_angle, utc_time

def get_Parallactic_Angle_Subaru(Az, El, lat=None, display = False):
    """
    Get the parallactic angle as coded in AO-188 and from the telemetry saved by it.
    Note, the parallactic given here is different from the one given by ``get_Parallactic_Angle''.
    It is not confirmed yet but here are two possible reasons: 
        * the numerical errors (Astroplan formula uses lots of trigonometry)
        * the levation calculated in AO-188 is not the exact same one given by Astroplan for the exact same time of observation
    
    :param Az: Azimuth of the object
    :type Az: float
    :param El: Elevation of the object
    :type El: float
    :param lat: latitude of the observing site. If None, the hard coded value in AO-188 (Subaru) is used, defaults to None
    :type lat: float, optional
    :param display: If True, displays the parallactic angle and the PAD (Parallactic angle in Chuck), defaults to False
    :type display: bool, optional 
    :return: Chuck PAD in degrees (Parallactic angle in Chuck coordinates base), parallactic angle as calculated in AO-188 in degree.
    :rtype: tuple

    """
    ## Formula from Kudo-san to compute the PAD on chuck from telescope AzEl info - from AO188 telemetry
    if lat is None:
        lat = 19.823806 * np.pi / 180
    else:
        lat = lat * np.pi / 180
    Az  = (Az) * np.pi / 180
    El  = El * np.pi / 180

    deg = np.sin(El) * np.cos(Az) + np.cos(El)*np.sin(lat)/np.cos(lat)
    para_angle_Subaru = np.arctan2(np.sin(Az), deg) * 180 / np.pi

    # if display is True:
    #     print('Parallactic angle for Subaru :'+str(para_angle_Subaru)+' degrees')

    ## Chuck's PAD 
    chuck_pad = para_angle_Subaru +180 - 39
    if display is True:
        print('Subaru parallactic angle for Chuck (PAD) :'+str(chuck_pad)+' degrees')

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
    inc         = np.deg2rad(inc)
    small_om    = np.deg2rad(small_om)
    big_om      = np.deg2rad(big_om)

    # Defining intermediate variables
    exq         = np.sqrt(1.-ecc*ecc)
    uu1         = np.cos(small_om)*np.cos(big_om)-np.sin(small_om)*np.sin(big_om)*np.cos(inc)
    uu2         = -1.*np.sin(small_om)*np.cos(big_om)-np.cos(small_om)*np.sin(big_om)*np.cos(inc)
    uu3         = np.cos(small_om)*np.sin(big_om)+np.sin(small_om)*np.cos(big_om)*np.cos(inc)
    uu4         = -1.*np.sin(small_om)*np.sin(big_om)+np.cos(small_om)*np.cos(big_om)*np.cos(inc)


    # Computing positions along the orbit
    u = 4.*np.pi*t
    x_pos    = (uu1*(np.cos(u)-ecc)*smaj+uu2*smaj*exq*np.sin(u))
    y_pos    = (uu3*(np.cos(u)-ecc)*smaj+uu4*smaj*exq*np.sin(u))
    
    return x_pos, y_pos
    
def binary_orbit(n_points, t0, T, smaj, ecc, inc, small_om, big_om, display = False):
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
    :return: epoch, x and y coordinates of the ``n_points'' positions on the orbit
    :rtype: tuple

    """

    x_pos       = np.zeros(n_points)
    y_pos       = np.zeros(n_points)
    epoch       = np.zeros(n_points)
    
    nn          = 2.*np.pi/T

    for i in range(n_points):
        t                   = i/(n_points-1)
        u                   = 4.*np.pi*t
        epoch[i]            = t0+(u-ecc*np.sin(u))/nn
        x_pos[i], y_pos[i]  = binary_orbit_calc(t, smaj, ecc, inc, small_om, big_om)

        
    # Plot orbit
    if display is True:
        plt.figure()
        plt.plot(y_pos, x_pos)
        plt.ylabel('Declination (arcsec)')
        plt.xlabel("Right ascension (arcsec)")
        plt.xlim([0.06,-0.06])
        plt.ylim([-0.06,0.06])
        plt.scatter(0,0,c='black',marker='+')
        plt.axis('equal')
        
    return epoch, x_pos, y_pos


def binary_ephem(current_date, t0, T, smaj, ecc, inc, small_om, big_om, display=False):
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
    
    date_sub = np.mod((current_date-t0), T) # Time spent after passing periastron
    # Solve equation to find the actual time locating the position of the companion
    x = sp.symbols('x', real=True)
    eq = sp.Eq(4*sp.pi*x-ecc*sp.sin(4*sp.pi*x), 2*sp.pi/T*date_sub)
    t = sp.nsolve(eq, x, 0)
    t = float(t)
    
    # Load orbit info
    x_date, y_date = binary_orbit_calc(t, smaj, ecc, inc, small_om, big_om)

    # Compute Sep. and Position Angle (PA)
    sep         = np.sqrt(x_date**2+y_date**2)
    pa          = np.rad2deg(np.arctan2(y_date,x_date))

    if pa < 0.:
        pa += 360.


    print('Binary separation :'+str(sep*1000)+' mas')
    print('Binary Position Angle :'+str(pa)+' degrees')

    x_pos, y_pos = binary_orbit(1000, t0, T, smaj, ecc, inc, small_om, big_om, display = False)[1:]
    
    if display is True :
        plt.figure()
        plt.plot(y_pos, x_pos)
        plt.plot(y_date,x_date,c='red',marker='+')
        plt.ylabel('Declination (arcsec)')
        plt.xlabel("Right ascension (arcsec)")
        plt.xlim([0.06,-0.06])
        plt.ylim([-0.06,0.06])
        plt.scatter(0,0,c='black',marker='+')
        plt.axis('equal')
        
    return sep, pa
    

def get_Orbit_Param(Target):
    """
    Provides orbit parameters for binary systems

    """


    if Target == 'Capella':
        # Time of Periastron [years]
        t0         = 1990.6984257357974
        # Semi-major axis [asec]
        smaj       = 0.056442
        # Eccentricity
        ecc        = 0.00089
        # Inclination [degrees]
        inc        = 137.156
        # Argument of periastron
        small_om   = 342.6
        # Position angle of ascending node
        big_om     = 40.522
        # Period [years]
        T          = 0.28479474332648874   

    elif  Target == 'betaHer':
        # Time of Periastron [years]
        t0 = Time('2415500.4',format='jd')
        t0         = t0.decimalyear
        # Semi-major axis [asec]
        smaj       = 0.01137
        # Eccentricity
        ecc        = 0.55
        # Inclination [degrees]
        inc        = 53.8
        # Argument of periastron
        small_om   = 24.6
        # Position angle of ascending node
        big_om     = 341.9
        # Period [years]
        T = 410.6/365.25

    elif Target == 'delSge':
        # Time of Periastron [years]
        t0         = 1979.93
        # Semi-major axis [asec]
        smaj       = 0.051
        # Eccentricity
        ecc        = 0.44
        # Inclination [degrees]
        inc        = 140
        # Argument of periastron
        small_om   = 257.7
        # Position angle of ascending node
        big_om     = 170.2
        # Period [years]
        T          = 10.11

    else:
        print('Target not listed...')



    return t0, smaj, ecc, inc, small_om, big_om, T





if __name__ == "__main__":

    # # Time of Periastron [years]
    # t0         = 1990.6984257357974
    # # Semi-major axis [asec]
    # smaj       = 0.056442
    # # Eccentricity
    # ecc        = 0.00089
    # # Inclination [degrees]
    # inc        = 137.156
    # # Argument of periastron
    # small_om   = 342.6
    # # Position angle of ascending node
    # big_om     = 40.522
    # # Period [years]
    # T          = 0.28479474332648874


    ## Load Orbits params
    t0, smaj, ecc, inc, small_om, big_om, T = get_Orbit_Param('Capella')

    ## Orbit    
    # epoch, x_pos, y_pos = binary_orbit(1000, t0, T, smaj, ecc, inc, small_om, big_om, display = False)


    date = Time('2021-03-19T00:00:00', format='isot')
    subaru = Observer.at_site("Subaru", timezone="US/Hawaii")

    ## Select date
    utc_time = subaru.datetime_to_astropy_time(date.datetime) # convert to UTC
    current_date = 2021.2123

    ## Compute / display binary ephem
    sep, pa = binary_ephem(current_date, t0, T, smaj, ecc, inc, small_om, big_om, display=True)
    

    ## Get the Parallactic Angle from astroplan library
    PA = get_Parallactic_Angle('Capella', 2020,9,16,4,48,32, display=True)
    ## Get the Parallactic Angle from Telescope position # WARNING : IF Az comes from AO188 telemetry, need to subtract 180 degrees before using get_Parallactic_Angle_Subaru()
    Az = 17.679
    El = 62.0691

    PAD, subaru_q = get_Parallactic_Angle_Subaru(Az, El, display = True)


    
    date = Time('2020-09-16T14:48:32', format='isot')    
    print('Seb', get_Parallactic_Angle('Capella', date.datetime.year, date.datetime.month, date.datetime.day, \
                                  date.datetime.hour-10, date.datetime.minute, date.datetime.second)[0])
        
    subaru = Observer.at_site("Subaru", timezone="US/Hawaii")
    star=FixedTarget.from_name("Capella")
    print(subaru.altaz(date, star).alt.deg, subaru.altaz(date, star).az.deg)
    print('Astroplan', subaru.parallactic_angle(date, star).deg)
    print('Subaru', get_Parallactic_Angle_Subaru(subaru.altaz(date, star).az.deg, subaru.altaz(date, star).alt.deg, display = False, lat=subaru.location.lat.deg)[1])
    
    LST = date.sidereal_time('mean', longitude=subaru.location.lon)
    H = (LST - star.ra).radian
    latitude = subaru.location.lat.radian
    declinaison = star.dec.radian
    sin_parallactic_angle = np.sin(H) * np.cos(latitude)
    cos_parallactic_angle = np.cos(declinaison) * np.sin(latitude) - np.sin(declinaison) * np.cos(latitude) * np.cos(H)
    parallactic_angle = np.arctan2(sin_parallactic_angle, cos_parallactic_angle) * 180/np.pi
    print('A la main', parallactic_angle)    
