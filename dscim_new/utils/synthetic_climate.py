"""
Synthetic climate data generation for testing and examples.

Generates GMST, GMSL, and FAIR climate data with realistic structure.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, List, Tuple


class ClimateDataGenerator:
    """
    Generate synthetic climate data for DSCIM workflows.

    Creates GMST (temperature), GMSL (sea level), FAIR temperature,
    FAIR GMSL, and pulse conversion data with realistic dimensions
    and structure.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default True
        Whether to print progress

    Examples
    --------
    >>> generator = ClimateDataGenerator(seed=42)
    >>> generator.generate_all_climate_data("climate_data")
    """

    def __init__(self, seed: Optional[int] = None, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def generate_gmst(
        self,
        years: Optional[np.ndarray] = None,
        rcps: Optional[List[str]] = None,
        gcms: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic GMST (Global Mean Surface Temperature) data.

        Parameters
        ----------
        years : np.ndarray, optional
            Years to generate (default: 2000-2100)
        rcps : list of str, optional
            RCP scenarios (default: ["rcp45", "rcp85"])
        gcms : list of str, optional
            GCM models (default: ["gcm1", "gcm2", "gcm3"])

        Returns
        -------
        pd.DataFrame
            GMST data with columns: year, rcp, gcm, anomaly

        Examples
        --------
        >>> gmst = generator.generate_gmst()
        """
        if years is None:
            years = np.arange(2000, 2101)
        if rcps is None:
            rcps = ["rcp45", "rcp85"]
        if gcms is None:
            gcms = ["gcm1", "gcm2", "gcm3"]

        self._log("Generating GMST data...")

        data = []
        for rcp in rcps:
            # RCP85 warms faster than RCP45
            warming_rate = 0.04 if rcp == "rcp85" else 0.025

            for gcm in gcms:
                # Each GCM has slightly different warming
                gcm_offset = np.random.normal(0, 0.2)

                for year in years:
                    # Temperature anomaly increases with time
                    baseline_anomaly = warming_rate * (year - 2000) + gcm_offset

                    # Add some noise
                    anomaly = baseline_anomaly + np.random.normal(0, 0.1)

                    data.append({
                        "year": year,
                        "rcp": rcp,
                        "gcm": gcm,
                        "anomaly": anomaly
                    })

        df = pd.DataFrame(data)
        self._log(f"Generated GMST: {len(df)} rows")
        return df

    def generate_gmsl(
        self,
        years: Optional[np.ndarray] = None,
        slr_models: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Generate synthetic GMSL (Global Mean Sea Level) data.

        Parameters
        ----------
        years : np.ndarray, optional
            Years to generate (default: 2000-2100)
        slr_models : list of str, optional
            Sea level rise models (default: ["slr1", "slr2", "slr3"])

        Returns
        -------
        xr.Dataset
            GMSL dataset with dims (year, slr) and variable 'gmsl'

        Examples
        --------
        >>> gmsl = generator.generate_gmsl()
        """
        if years is None:
            years = np.arange(2000, 2101)
        if slr_models is None:
            slr_models = ["slr1", "slr2", "slr3"]

        self._log("Generating GMSL data...")

        n_years = len(years)
        n_slr = len(slr_models)

        # Generate sea level rise (meters)
        # Starts near 0, increases over time
        gmsl_data = np.zeros((n_years, n_slr))

        for i, slr_model in enumerate(slr_models):
            # Each model has different rise rate
            rise_rate = 0.003 + np.random.uniform(-0.001, 0.001)

            for j, year in enumerate(years):
                # Quadratic rise over time
                years_elapsed = year - years[0]
                gmsl = rise_rate * years_elapsed + 0.0001 * years_elapsed**2

                # Add noise
                gmsl += np.random.normal(0, 0.01)

                gmsl_data[j, i] = gmsl

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                "gmsl": (["year", "slr"], gmsl_data)
            },
            coords={
                "year": years,
                "slr": slr_models
            }
        )

        ds.gmsl.attrs["units"] = "meters"
        ds.gmsl.attrs["long_name"] = "Global Mean Sea Level"

        self._log(f"Generated GMSL: {ds.gmsl.shape}")
        return ds

    def generate_fair_temperature(
        self,
        years: Optional[np.ndarray] = None,
        rcps: Optional[List[str]] = None,
        simulations: Optional[List[int]] = None,
        gases: Optional[List[str]] = None,
        pulse_years: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """
        Generate synthetic FAIR temperature data (control and pulse).

        Parameters
        ----------
        years : np.ndarray, optional
            Years (default: 2000-2300)
        rcps : list of str, optional
            RCP scenarios
        simulations : list of int, optional
            Simulation indices (default: 0-9)
        gases : list of str, optional
            Greenhouse gases (default: ["CO2", "CH4"])
        pulse_years : list of int, optional
            Pulse years (default: [2020])

        Returns
        -------
        xr.Dataset
            FAIR temperature with control_temperature and pulse_temperature

        Examples
        --------
        >>> fair_temp = generator.generate_fair_temperature()
        """
        if years is None:
            years = np.arange(2000, 2301)
        if rcps is None:
            rcps = ["rcp45", "rcp85"]
        if simulations is None:
            simulations = list(range(10))
        if gases is None:
            gases = ["CO2", "CH4"]
        if pulse_years is None:
            pulse_years = [2020]

        self._log("Generating FAIR temperature data...")

        n_years = len(years)
        n_rcp = len(rcps)
        n_sim = len(simulations)
        n_gas = len(gases)
        n_pulse = len(pulse_years)

        # Control temperature (no pulse)
        control_shape = (n_years, n_rcp, n_sim)
        control_temp = np.zeros(control_shape)

        for i_rcp, rcp in enumerate(rcps):
            warming_rate = 0.04 if rcp == "rcp85" else 0.025

            for i_sim in range(n_sim):
                sim_offset = np.random.normal(0, 0.3)

                for i_year, year in enumerate(years):
                    temp = warming_rate * (year - 2000) + sim_offset
                    temp += np.random.normal(0, 0.15)
                    control_temp[i_year, i_rcp, i_sim] = temp

        # Pulse temperature (with pulse)
        pulse_shape = (n_years, n_rcp, n_sim, n_gas, n_pulse)
        pulse_temp = np.zeros(pulse_shape)

        for i_pulse, pulse_year in enumerate(pulse_years):
            pulse_idx = np.where(years == pulse_year)[0][0] if pulse_year in years else len(years)//2

            for i_gas, gas in enumerate(gases):
                # CO2 has larger, longer-lasting effect than CH4
                if gas == "CO2":
                    pulse_magnitude = 0.05
                    decay_rate = 0.001
                else:  # CH4
                    pulse_magnitude = 0.02
                    decay_rate = 0.01

                for i_rcp in range(n_rcp):
                    for i_sim in range(n_sim):
                        for i_year, year in enumerate(years):
                            # Start with control temperature
                            temp = control_temp[i_year, i_rcp, i_sim]

                            # Add pulse effect after pulse year
                            if i_year >= pulse_idx:
                                years_since_pulse = year - pulse_year
                                pulse_effect = pulse_magnitude * np.exp(-decay_rate * years_since_pulse)
                                temp += pulse_effect

                            pulse_temp[i_year, i_rcp, i_sim, i_gas, i_pulse] = temp

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                "control_temperature": (["year", "rcp", "simulation"], control_temp),
                "pulse_temperature": (["year", "rcp", "simulation", "gas", "pulse_year"], pulse_temp),
            },
            coords={
                "year": years,
                "rcp": rcps,
                "simulation": simulations,
                "gas": gases,
                "pulse_year": pulse_years,
            }
        )

        ds.control_temperature.attrs["units"] = "degrees_C"
        ds.pulse_temperature.attrs["units"] = "degrees_C"

        self._log(f"Generated FAIR temperature: control={control_temp.shape}, pulse={pulse_temp.shape}")
        return ds

    def generate_fair_gmsl(
        self,
        years: Optional[np.ndarray] = None,
        slr_models: Optional[List[str]] = None,
        simulations: Optional[List[int]] = None,
        gases: Optional[List[str]] = None,
        pulse_years: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """
        Generate synthetic FAIR GMSL data (control and pulse).

        Parameters
        ----------
        years : np.ndarray, optional
            Years (default: 2000-2300)
        slr_models : list of str, optional
            Sea level models
        simulations : list of int, optional
            Simulation indices
        gases : list of str, optional
            Greenhouse gases
        pulse_years : list of int, optional
            Pulse years

        Returns
        -------
        xr.Dataset
            FAIR GMSL with control_gmsl and pulse_gmsl

        Examples
        --------
        >>> fair_gmsl = generator.generate_fair_gmsl()
        """
        if years is None:
            years = np.arange(2000, 2301)
        if slr_models is None:
            slr_models = ["slr1", "slr2", "slr3"]
        if simulations is None:
            simulations = list(range(10))
        if gases is None:
            gases = ["CO2", "CH4"]
        if pulse_years is None:
            pulse_years = [2020]

        self._log("Generating FAIR GMSL data...")

        n_years = len(years)
        n_slr = len(slr_models)
        n_sim = len(simulations)
        n_gas = len(gases)
        n_pulse = len(pulse_years)

        # Control GMSL
        control_shape = (n_years, n_slr, n_sim)
        control_gmsl = np.zeros(control_shape)

        for i_slr in range(n_slr):
            rise_rate = 0.003 + np.random.uniform(-0.001, 0.001)

            for i_sim in range(n_sim):
                sim_offset = np.random.normal(0, 0.01)

                for i_year, year in enumerate(years):
                    years_elapsed = year - years[0]
                    gmsl = rise_rate * years_elapsed + 0.00005 * years_elapsed**2 + sim_offset
                    gmsl += np.random.normal(0, 0.005)
                    control_gmsl[i_year, i_slr, i_sim] = gmsl

        # Pulse GMSL
        pulse_shape = (n_years, n_slr, n_sim, n_gas, n_pulse)
        pulse_gmsl = np.zeros(pulse_shape)

        for i_pulse, pulse_year in enumerate(pulse_years):
            pulse_idx = np.where(years == pulse_year)[0][0] if pulse_year in years else len(years)//2

            for i_gas, gas in enumerate(gases):
                if gas == "CO2":
                    pulse_magnitude = 0.001
                    decay_rate = 0.0005
                else:
                    pulse_magnitude = 0.0005
                    decay_rate = 0.005

                for i_slr in range(n_slr):
                    for i_sim in range(n_sim):
                        for i_year, year in enumerate(years):
                            gmsl = control_gmsl[i_year, i_slr, i_sim]

                            if i_year >= pulse_idx:
                                years_since_pulse = year - pulse_year
                                pulse_effect = pulse_magnitude * np.exp(-decay_rate * years_since_pulse)
                                gmsl += pulse_effect

                            pulse_gmsl[i_year, i_slr, i_sim, i_gas, i_pulse] = gmsl

        # Create dataset - note dimension name is 'pulse_years' (plural) for GMSL
        ds = xr.Dataset(
            {
                "control_gmsl": (["year", "slr", "simulation"], control_gmsl),
                "pulse_gmsl": (["year", "slr", "simulation", "gas", "pulse_years"], pulse_gmsl),
            },
            coords={
                "year": years,
                "slr": slr_models,
                "simulation": simulations,
                "gas": gases,
                "pulse_years": pulse_years,  # Note: plural
            }
        )

        ds.control_gmsl.attrs["units"] = "meters"
        ds.pulse_gmsl.attrs["units"] = "meters"

        self._log(f"Generated FAIR GMSL: control={control_gmsl.shape}, pulse={pulse_gmsl.shape}")
        return ds

    def generate_pulse_conversion(
        self,
        gases: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Generate pulse conversion factors.

        Converts large pulses (gigatons) to one-ton pulse equivalents.

        Parameters
        ----------
        gases : list of str, optional
            Greenhouse gases

        Returns
        -------
        xr.Dataset
            Conversion factors with variable 'emissions'

        Examples
        --------
        >>> conversion = generator.generate_pulse_conversion()
        """
        if gases is None:
            gases = ["CO2", "CH4"]

        self._log("Generating pulse conversion factors...")

        # Conversion factors (arbitrary but realistic)
        conversion_factors = {
            "CO2": 1.0,  # 1 GtC
            "CH4": 0.1,  # CH4 pulses are smaller
        }

        emissions = np.array([conversion_factors.get(gas, 1.0) for gas in gases])

        ds = xr.Dataset(
            {
                "emissions": (["gas"], emissions)
            },
            coords={
                "gas": gases
            }
        )

        ds.emissions.attrs["units"] = "GtC"
        ds.emissions.attrs["long_name"] = "Pulse conversion factor"

        self._log(f"Generated pulse conversion: {len(gases)} gases")
        return ds

    def generate_all_climate_data(
        self,
        output_dir: str,
        years_gmst: Optional[np.ndarray] = None,
        years_fair: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Generate and save all climate data files.

        Parameters
        ----------
        output_dir : str
            Output directory for climate data
        years_gmst : np.ndarray, optional
            Years for GMST/GMSL (default: 2000-2100)
        years_fair : np.ndarray, optional
            Years for FAIR data (default: 2000-2300)

        Returns
        -------
        dict
            Paths to all generated files

        Examples
        --------
        >>> generator = ClimateDataGenerator(seed=42)
        >>> paths = generator.generate_all_climate_data("climate_data")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if years_gmst is None:
            years_gmst = np.arange(2000, 2101)
        if years_fair is None:
            years_fair = np.arange(2000, 2301)

        paths = {}

        # GMST
        self._log("\n=== Generating GMST ===")
        gmst = self.generate_gmst(years=years_gmst)
        gmst_path = output_path / "GMTanom_all_temp.csv"
        gmst.to_csv(gmst_path, index=False)
        paths["gmst"] = str(gmst_path)
        self._log(f"Saved: {gmst_path}")

        # GMSL
        self._log("\n=== Generating GMSL ===")
        gmsl = self.generate_gmsl(years=years_gmst)
        gmsl_path = output_path / "coastal_gmsl.zarr"
        gmsl.to_zarr(gmsl_path, mode='w')
        paths["gmsl"] = str(gmsl_path)
        self._log(f"Saved: {gmsl_path}")

        # FAIR Temperature
        self._log("\n=== Generating FAIR Temperature ===")
        fair_temp = self.generate_fair_temperature(years=years_fair)
        fair_temp_path = output_path / "ar6_fair162_sim.nc"
        fair_temp.to_netcdf(fair_temp_path)
        paths["fair_temperature"] = str(fair_temp_path)
        self._log(f"Saved: {fair_temp_path}")

        # FAIR GMSL
        self._log("\n=== Generating FAIR GMSL ===")
        fair_gmsl = self.generate_fair_gmsl(years=years_fair)
        fair_gmsl_path = output_path / "scenario_gmsl.nc4"
        fair_gmsl.to_netcdf(fair_gmsl_path)
        paths["fair_gmsl"] = str(fair_gmsl_path)
        self._log(f"Saved: {fair_gmsl_path}")

        # Pulse Conversion
        self._log("\n=== Generating Pulse Conversion ===")
        conversion = self.generate_pulse_conversion()
        conversion_path = output_path / "conversion.nc4"
        conversion.to_netcdf(conversion_path)
        paths["pulse_conversion"] = str(conversion_path)
        self._log(f"Saved: {conversion_path}")

        self._log(f"\n✓ All climate data generated in: {output_dir}")
        return paths
