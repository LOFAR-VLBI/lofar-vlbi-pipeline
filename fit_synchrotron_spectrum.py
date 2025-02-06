import argparse
from abc import ABC, abstractmethod
from typing import Any
import os

import astropy
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pyvo
from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier
from scipy.optimize import curve_fit

plt.rcParams.update(
    {
        # "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.size": 20,
    }
)


class Fitter(ABC):
    fit_parameters = None
    fit_parameters_chi = None
    freq_points = 0

    def __init__(self, freq_ref=144e6):
        self.freq_ref = freq_ref

    @abstractmethod
    def fit_func(self, *args, **kwargs) -> Any:
        pass

    def fit(self, freq: np.ndarray, flux_density: np.ndarray, p0: tuple, sigma: np.ndarray) -> tuple:
        self.freq_points = len(freq)
        popt, pcov = curve_fit(self.fit_func, xdata=freq, ydata=flux_density, p0=p0, sigma=sigma)
        self.fit_parameters = popt
        residuals = flux_density - self.fit_func(np.array(freq), *popt)
        flux_err = sigma
        chi_sqr = np.sum(((residuals)/flux_err)**2)
        self.stats = chi_sqr
        return popt, chi_sqr
    
    def plot(
        self,
        freqs,
        fluxs,
        f_err,
        point_labels: list[str] | None = None,
        file_name: str | None = None,
        outdir=".",
    ):
        freq_smooth = np.linspace(10e6, 10e9, 1000)
        plt.figure(figsize=(8, 8))
        plt.loglog()
        plt.gca().set(xlabel="Frequency [Hz]", ylabel="Flux density [Jy]")
        plt.setp(plt.gca().spines.values(), linewidth=1.5)
        plt.gca().xaxis.set_tick_params(width=1.5, which="both")
        plt.gca().yaxis.set_tick_params(width=1.5, which="both")
        plt.errorbar(freqs, fluxs, yerr=f_err, ls='none')
        plt.scatter(freqs, fluxs, marker="s", ec="k", fc="gray", s=64)

        if point_labels:
            for f, s, l in zip(freqs, fluxs, point_labels):
                plt.annotate(l, (f, s * 1.1), fontsize=14)
        plt.plot(
            freq_smooth,
            self.fit_func(freq_smooth, *self.fit_parameters),
            color="k",
            linestyle="--",
            linewidth=1.5,
        )
        # plt.text(0.1, 0.2, "Spectral fit:\n\n${:.2f} \\left(\\frac{{\\nu}}{{{:.0f}\\ \\mathrm{{MHz}}}}\\right)^{{{:.2f} + {:.2f} \\log_{{10}}(\\nu / {:.0f}\\ \\mathrm{{MHz}})}}$".format(self.fit_parameters[0], self.freq_ref/1e6, self.fit_parameters[1], self.fit_parameters[2], self.freq_ref/1e6), transform=plt.gca().transAxes, fontsize=16)
        plt.text(
            0.1,
            0.2,
            "Fitted parameters:\nS = {:.2f}\n$\\alpha_1$ = {:.2f}\n$\\alpha_2$ = {:.2f}\n$\\chi^2$ = {:.2f}\n$\\nu_{{\\mathrm{{ref}}}}$ = {:.2f} MHz".format(
                self.fit_parameters[0],
                self.fit_parameters[1],
                self.fit_parameters[2],
                self.stats,
                self.freq_ref / 1e6,
            ),
            transform=plt.gca().transAxes,
            fontsize=16,
            va="center_baseline",
        )
        plt.xlim(10e6, 10e9)
        plt.ylim(10e-3, 10)
        if file_name:
            plt.savefig(
                os.path.join(outdir, f"{file_name}.png"), dpi=300, bbox_inches="tight"
            )
        else:
            plt.savefig(
                os.path.join(outdir, "spectrum.png"), dpi=300, bbox_inches="tight"
            )


class LogFitter(Fitter):
    def fit_func(self, freq, I0, alpha, beta) -> float | np.ndarray:
        return I0 * (freq / self.freq_ref) ** (
            alpha + beta * np.log10(freq / self.freq_ref)
        )


def fit_from_NED(ra: float, dec: float, radius: float, outdir: str):
    obj = Ned.query_region(f"{ra}d {dec}d", radius=radius * u.arcsec)["Object Name"][0]

    ned_table = Ned.get_table(obj, table="photometry")
    ned_photometry = ned_table[np.where(ned_table["Frequency"] < 1e10)]
    freqs = ned_photometry["Frequency"]
    fluxd_ned = ned_photometry["Flux Density"]
    fluxd_Ned_err = 0.1 * fluxd_ned # PLACEHOLDER
    fitter = LogFitter()
    fitter.fit(freqs, fluxd_ned, p0=(1.0, -0.8, 0.0), sigma=fluxd_Ned_err)
    fitter.plot(
        freqs, fluxd_ned, file_name=f"spectrum_{ra:.3f}_{dec:.3f}_NED", outdir=outdir
    )
    return fitter


def query_vizier(
    catalogue: str, ra: float, dec: float, radius: float
) -> astropy.table.table.Table:
    v = Vizier(catalog=catalogue)
    q = v.query_region(f"{ra}, {dec}", radius=radius * u.arcsec)
    if q:
        return q[0]
    else:
        raise RuntimeError("Source not found in requested survey.")


def query_bootstrap(
    catalogue: str, ra: float, dec: float, radius: float
) -> astropy.table.table.Table:
    with fits.open(catalogue) as hdul:
        catalog = astropy.table.Table(hdul[1].data)
    cat_coords = SkyCoord(catalog["RA"], catalog["DEC"], unit="deg")
    targ_coord = SkyCoord(ra, dec, unit="deg")
    sep = targ_coord.separation(cat_coords)
    match = sep < (radius * u.arcsec)
    q = catalog[match]
    if q:
        return q[0]
    else:
        raise RuntimeError("Source not found in requested survey.")


def query_vo(
    vo_server: str, ra: float, dec: float, radius: float
) -> pyvo.dal.scs.SCSResults:
    query = pyvo.dal.scs.SCSQuery(vo_server, maxrec=10)
    query["RA"] = ra
    query["DEC"] = dec
    query.radius = radius
    t = query.execute()
    return t


def fit_from_trusted_surveys(ra: float, dec: float, radius: float, outdir: str):
    frequency = []
    flux_density = []
    flux_err = []
    survey_name = []
    has_survey = 0b0000000
    try:
        s_lolss = (
            query_vizier(VIZIER_LOLSS_DR1, ra, dec, radius)["Ftot"].to("Jy").value[0]
        )
        has_survey ^= 0b1000000
        frequency.append(60e6)
        flux_density.append(s_lolss)
        flux_err.append(0.1*s_lolss) #PLACEHOLDER
        survey_name.append("LOLSS")
    except RuntimeError:
        print("Source not in LoLSS DR1")


    # Old VLSSr code which we think is wrong
    # try:
    #     s_vlssr = query_vizier(VIZIER_VLSSr, ra, dec, radius)["Sp"].to("Jy").value[0]
    #     has_survey ^= 0b0100000
    #     frequency.append(74e6)
    #     flux_density.append(s_vlssr)
    #     survey_name.append("VLSSr")
    # except RuntimeError:
    #     print("Source not in VLSSr")

    try:
        s_vlssr = query_bootstrap(BOOTSTRAP_VSSLr, ra, dec, radius)["Total_flux"] # Jy 
        e_s_vlssr = query_bootstrap(BOOTSTRAP_VSSLr, ra, dec, radius)["E_Total_flux"] # Jy
        frequency.append(74e6)
        flux_density.append(s_vlssr)
        flux_err.append(e_s_vlssr) 
        survey_name.append("VLSSr")
    except RuntimeError:
        print("Source not in VLSSr")

    try:
        lotss = query_vo(VO_LOTSS_DR2, ra, dec, 3.0 / 3600)
        s_lotss = lotss["Total_flux"][0] * 1e-3
        e_s_lotss = lotss["E_Total_flux"][0] * 1e-3
        has_survey ^= 0b0010000
        frequency.append(144e6)
        survey_name.append("LOTSS")
        flux_density.append(s_lotss)
        flux_err.append(e_s_lotss) 
        HAS_LOTSS = True
    except RuntimeError:
        HAS_LOTSS = False
        print("Source not in LOTSS, trying TGSS")
        try:
            s_tgss = (
                query_vizier(VIZIER_TGSS, ra, dec, radius)["Stotal"].to("Jy").value[0]
            )
            has_survey ^= 0b0010000
            frequency.append(150e6)
            flux_density.append(s_tgss)
            flux_err.append(0.1*s_tgss) #PLACEHOLDER
            survey_name.append("TGSS")
        except RuntimeError:
            print("Source not in TGSS")
    HAS_TGSS = not HAS_LOTSS

    try:
        s_wenss = query_vizier(VIZIER_WENSS, ra, dec, radius)["Sint"].to("Jy").value[0]
        has_survey ^= 0b0001000
        # 325 MHz in main part, 352 MHz in polar part
        if dec < 72:
            frequency.append(325e6)
        else:
            frequency.append(352e6)
        flux_density.append(s_wenss)
        flux_err.append(0.1*s_wenss) #PLACEHOLDER
        survey_name.append("WENSS")
    except RuntimeError:
        print("Source not in WENSS")

    try:
        s_sumss = query_vizier(VIZIER_SUMSS, ra, dec, radius)["Sint"].to("Jy").value[0]
        has_survey ^= 0b0000100
        frequency.append(843e6)
        flux_density.append(s_sumss)
        flux_err.append(0.1*s_sumss) #PLACEHOLDER
        survey_name.append("SUMSS")
    except RuntimeError:
        print("Source not in SUMSS")

    try:
        s_first = query_vizier(VIZIER_FIRST, ra, dec, radius)["Fint"].to("Jy").value[0]
        has_survey ^= 0b0000010
        frequency.append(1.4e9)
        flux_density.append(s_first)
        flux_err.append(0.1*s_first)
        survey_name.append("FIRST")
    except RuntimeError:
        print("Source not in FIRST")

    try:
        s_gb6 = query_vizier(VIZIER_GB6, ra, dec, radius)["Flux"].to("Jy").value[0]
        has_survey ^= 0b0000001
        frequency.append(4.85e9)
        flux_density.append(s_gb6)
        flux_err.append(0.1*s_gb6) #PLACEHOLDER
        survey_name.append("GB6")
    except RuntimeError:
        print("Source not in GB6")

    # try:
    #    s_vlass = query_vizier(VIZIER_VLASS_QL1, ra, dec, radius)["Ftot"].to("Jy").value[0]
    #    #has_survey ^= 0b0000001
    #    frequency.append(3e9)
    #    # VLASS reports a ~15% underestimate of measurements in the QL catalogues,
    #    # so we roughly correct that here.
    #    flux_density.append(s_vlass)# * 1.15)
    #    flux_err.append(0.1*flux_density) #PLACEHOLDER
    #    survey_name.append("VLASS")
    # except RuntimeError:
    #    print("Source not in VLASS QL1")
    print(f"Survey mask: {has_survey:07b}")
    # if HAS_LOTSS:
    #     fitter = LogFitter(freq_ref=144e6)
    # elif HAS_TGSS:
    #     fitter = LogFitter(freq_ref=150e6)

    fitter = LogFitter(freq_ref=144e6)
    fitter.fit(frequency, flux_density, p0=(1.0, -0.8, 0.0), sigma=flux_err)
    fitter.plot(
        frequency,
        flux_density,
        flux_err,
        point_labels=survey_name,
        file_name=f"spectrum_{ra:.3f}_{dec:.3f}",
        outdir=outdir,
    )
    return fitter


# 55 MHz
VIZIER_LOLSS_DR1 = "J/A+A/673/A165/lolss1g"
# 74 MHz - ???
VIZIER_VLSSr = "VIII/97/catalog"
# VLSSr_FIXED
BOOTSTRAP_VSSLr = "https://www.extragalactic.info/bootstrap/VLSS.fits"
# 325 MHz
VIZIER_WENSS = "VIII/62/wenss"
# 1.4 GHz
VIZIER_FIRST = "VIII/92/first14"
# 4.85 GHz
VIZIER_GB6 = "VIII/40/gb6"
# 843 MHz
VIZIER_SUMSS = "VIII/81B/sumss212"
# 150 MHz
VIZIER_TGSS = "J/A+A/598/A78/table3"
# 3 GHz
VIZIER_VLASS_QL1 = "J/ApJS/255/30/comp"

# 144 MHz
VO_LOTSS_DR2 = "https://vo.astron.nl/lotss_dr2/q/src_cone/scs.xml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a synchrotron spectrum for a specific source using archival radio data."
    )
    parser.add_argument(
        "--output_dir",
        dest="outdir",
        type=str,
        help="directory to save results in [default cwd]",
        default=".",
    )
    parser.add_argument(
        "--ra", type=float, help="Right ascension of the target source."
    )
    parser.add_argument("--dec", type=float, help="Declination of the target source.")
    parser.add_argument(
        "--match_radius",
        type=float,
        help="Radius in arcsec within which to allow matching with archival sources (N.B. some surveys have a hardcoded matchin radius).",
        default=12.0,
    )
    parser.add_argument(
        "--photometry_mode",
        choices=["trusted", "NED"],
        type=str,
        help="Set to 'trusted' to only use vetted radio surveys for fitting the spectrum. Set to 'NED' to use whatever photometry is available in NED between 10 MHz and 10 GHz.",
        default="trusted",
    )
    args = parser.parse_args()
    if args.photometry_mode == "trusted":
        fit_from_trusted_surveys(args.ra, args.dec, args.match_radius)
    elif args.photometry_mode == "NED":
        fit_from_NED(args.ra, args.dec, args.match_radius)
