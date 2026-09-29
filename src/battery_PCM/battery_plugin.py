"""Prescient plugin: a rule-based battery that trades only in the real-time market.

How it works
------------
1. The day-ahead market (RUC + DA pricing) is solved WITHOUT the battery. This
   plugin never touches the RUC.
2. Before every real-time economic dispatch (SCED) is solved, the battery looks
   up the day-ahead LMP at its bus for that hour and applies a price rule:

       DA LMP <  buy_price   -> charge    (buy from the grid, bus load goes up)
       DA LMP >  sell_price  -> discharge (sell to the grid, bus load goes down)
       otherwise             -> idle

   Charging / discharging power is capped by the power ratings and by what the
   state of charge (SoC) allows in that interval.
3. The battery is injected into the SCED as its own load element
   ("BATTERY_<bus name>") at its bus: positive p_load = charging, negative
   p_load = discharging. The bus's own load is left untouched, so Prescient's
   bus "Demand" output is the original load plus the battery's net load.
4. After the SCED is solved, the SoC is updated and the step is written to
   <output_directory>/battery_results.csv, including the real-time LMP at the
   bus and the battery's real-time revenue (the battery holds no DA position,
   so it settles fully at the RT LMP).

The battery is a price taker that self-schedules: the SCED must serve its load,
so the dispatched power always equals what the rule asked for.

Usage (from a run script)
-------------------------
    prescient_options["plugin"] = {
        "battery": {
            "module": "path/to/battery_plugin.py",   # or the imported module
            "battery_bus": "Abel",                   # Bus Name or Bus ID ("101")
            "battery_p_charge_max": 50.0,            # MW
            "battery_p_discharge_max": 50.0,         # MW
            "battery_soc_max": 200.0,                # MWh
        }
    }

Requires "compute_market_settlements": True, otherwise Prescient does not solve
the day-ahead pricing problem and there are no DA LMPs to read.
"""

import csv
import os

from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat

# column order of battery_results.csv
RESULT_COLUMNS = [
    "Date",
    "Hour",
    "Minute",
    "Bus",
    "DA LMP [$/MWh]",
    "RT LMP [$/MWh]",
    "Action",
    "Charge [MW]",
    "Discharge [MW]",
    "Net Load [MW]",
    "SoC Start [MWh]",
    "SoC End [MWh]",
    "RT Revenue [$]",
]

# power below this (MW) is treated as zero, to avoid dust from SoC round-off
_TOL = 1e-6


def _unit_interval(value):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {value}")
    return value


def get_configuration(key):
    """Declare the battery options. They are set through the plugin dict in the
    Prescient options, or on the command line as --battery-bus etc."""
    config = ConfigDict()

    config.declare("battery_bus", ConfigValue(
        domain=str, default=None,
        description="Bus the battery connects to: the RTS-GMLC Bus Name "
                    "(e.g. 'Abel') or Bus ID (e.g. '101').",
    )).declare_as_argument()

    config.declare("battery_p_charge_max", ConfigValue(
        domain=NonNegativeFloat, default=None,
        description="Maximum charging power drawn from the grid [MW].",
    )).declare_as_argument()

    config.declare("battery_p_discharge_max", ConfigValue(
        domain=NonNegativeFloat, default=None,
        description="Maximum discharging power injected to the grid [MW].",
    )).declare_as_argument()

    config.declare("battery_soc_max", ConfigValue(
        domain=NonNegativeFloat, default=None,
        description="Maximum state of charge, i.e. energy capacity [MWh].",
    )).declare_as_argument()

    config.declare("battery_soc_min", ConfigValue(
        domain=NonNegativeFloat, default=0.0,
        description="Minimum state of charge [MWh].",
    )).declare_as_argument()

    config.declare("battery_soc_init", ConfigValue(
        domain=NonNegativeFloat, default=None,
        description="State of charge at the start of the simulation [MWh]. "
                    "Defaults to battery_soc_min.",
    )).declare_as_argument()

    config.declare("battery_charge_efficiency", ConfigValue(
        domain=_unit_interval, default=1.0,
        description="Charging efficiency: SoC gains eta_c * P_charge * dt.",
    )).declare_as_argument()

    config.declare("battery_discharge_efficiency", ConfigValue(
        domain=_unit_interval, default=1.0,
        description="Discharging efficiency: SoC loses P_discharge * dt / eta_d.",
    )).declare_as_argument()

    config.declare("battery_buy_price", ConfigValue(
        domain=float, default=10.0,
        description="Charge when the DA LMP at the bus is below this [$/MWh].",
    )).declare_as_argument()

    config.declare("battery_sell_price", ConfigValue(
        domain=float, default=15.0,
        description="Discharge when the DA LMP at the bus is above this [$/MWh].",
    )).declare_as_argument()

    config.declare("battery_results_file", ConfigValue(
        domain=str, default="battery_results.csv",
        description="Name of the per-step results CSV, written in the "
                    "Prescient output directory.",
    )).declare_as_argument()

    return config


class RuleBasedBattery:
    """Holds the battery's parameters and state, and implements the callbacks."""

    def __init__(self, plugin_config):
        c = plugin_config
        missing = [name for name in ("battery_bus", "battery_p_charge_max",
                                     "battery_p_discharge_max", "battery_soc_max")
                   if c[name] is None]
        if missing:
            raise ValueError(f"battery plugin: these options must be set: {missing}")

        self.bus_spec = str(c.battery_bus)
        self.p_charge_max = float(c.battery_p_charge_max)
        self.p_discharge_max = float(c.battery_p_discharge_max)
        self.soc_max = float(c.battery_soc_max)
        self.soc_min = float(c.battery_soc_min)
        self.eta_c = float(c.battery_charge_efficiency)
        self.eta_d = float(c.battery_discharge_efficiency)
        self.buy_price = float(c.battery_buy_price)
        self.sell_price = float(c.battery_sell_price)
        self.results_file_name = c.battery_results_file

        soc_init = c.battery_soc_init
        self.soc = self.soc_min if soc_init is None else float(soc_init)

        if self.soc_min > self.soc_max:
            raise ValueError("battery_soc_min must not exceed battery_soc_max")
        if not self.soc_min <= self.soc <= self.soc_max:
            raise ValueError(
                f"battery_soc_init ({self.soc}) must lie in "
                f"[battery_soc_min, battery_soc_max] = [{self.soc_min}, {self.soc_max}]")
        if self.buy_price > self.sell_price:
            raise ValueError("battery_buy_price must not exceed battery_sell_price")

        self.bus = None          # Egret bus name, resolved on the first SCED
        self.load_name = None    # name of the battery's load element
        self.results_path = None
        self._pending = None     # the plan for the SCED being solved
        self._totals = {"charged_MWh": 0.0, "discharged_MWh": 0.0, "revenue": 0.0}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _resolve_bus(self, md):
        """Map the configured bus (name or ID) to the Egret bus name."""
        buses = md.data["elements"]["bus"]
        if self.bus_spec in buses:
            return self.bus_spec
        for name, bus_dict in buses.items():
            if str(bus_dict.get("id")) == self.bus_spec:
                return name
        raise ValueError(
            f"battery plugin: bus '{self.bus_spec}' matches neither a bus name "
            f"nor a bus ID in the network")

    def _da_lmp(self, simulator, options, period_start_minutes):
        """DA LMP at the battery's bus for the hour a SCED period falls in.

        period_start_minutes is measured from the top of the current hour.
        Returns None if the active DA market has no price for that hour (e.g. a
        look-ahead period beyond the RUC horizon).
        """
        ruc_market = simulator.data_manager.ruc_market_active
        if ruc_market is None:
            raise RuntimeError(
                "battery plugin: no active day-ahead market. Set "
                "'compute_market_settlements': True in the Prescient options.")
        hour = simulator.time_manager.current_time.hour
        # same index Prescient uses for DA settlement (hour % ruc_every_hours),
        # shifted by how many whole hours ahead the period starts
        index = hour % options.ruc_every_hours + int(period_start_minutes // 60)
        return ruc_market.day_ahead_prices.get((self.bus, index))

    def _decide(self, da_lmp, soc, dt):
        """Apply the price rule. Returns (action, p_charge, p_discharge)."""
        if da_lmp is None:
            return "idle", 0.0, 0.0
        if da_lmp < self.buy_price:
            headroom = max(self.soc_max - soc, 0.0)
            p = min(self.p_charge_max, headroom / (self.eta_c * dt))
            if p > _TOL:
                return "charge", p, 0.0
        elif da_lmp > self.sell_price:
            available = max(soc - self.soc_min, 0.0)
            p = min(self.p_discharge_max, available * self.eta_d / dt)
            if p > _TOL:
                return "discharge", 0.0, p
        return "idle", 0.0, 0.0

    def _next_soc(self, soc, p_charge, p_discharge, dt):
        soc = soc + self.eta_c * p_charge * dt - p_discharge * dt / self.eta_d
        # clip round-off
        return min(max(soc, self.soc_min), self.soc_max)

    # ------------------------------------------------------------------ #
    # Prescient callbacks
    # ------------------------------------------------------------------ #
    def initialize(self, options, simulator):
        if not options.compute_market_settlements:
            raise RuntimeError(
                "battery plugin needs DA LMPs: set 'compute_market_settlements': "
                "True in the Prescient options.")

        os.makedirs(options.output_directory, exist_ok=True)
        self.results_path = os.path.join(options.output_directory,
                                         self.results_file_name)
        with open(self.results_path, "w", newline="") as f:
            csv.writer(f).writerow(RESULT_COLUMNS)

        print(f"[battery] bus={self.bus_spec}, P_ch={self.p_charge_max} MW, "
              f"P_dis={self.p_discharge_max} MW, SoC in [{self.soc_min}, "
              f"{self.soc_max}] MWh, SoC0={self.soc} MWh, buy<{self.buy_price}, "
              f"sell>{self.sell_price} $/MWh")

    def before_sced(self, options, simulator, sced_instance):
        """Put the battery's planned net load into the SCED about to be solved."""
        if self.bus is None:
            self.bus = self._resolve_bus(sced_instance)
            self.load_name = f"BATTERY_{self.bus}"
            print(f"[battery] connected at bus '{self.bus}' as load '{self.load_name}'")

        system = sced_instance.data["system"]
        n_periods = len(system["time_keys"])
        step_minutes = system["time_period_length_minutes"]
        dt = step_minutes / 60.0
        minute = simulator.time_manager.current_time.datetime.minute

        # plan every SCED period, projecting SoC through the look-ahead
        soc = self.soc
        plan = []
        for t in range(n_periods):
            da_lmp = self._da_lmp(simulator, options, minute + t * step_minutes)
            action, p_c, p_d = self._decide(da_lmp, soc, dt)
            plan.append((da_lmp, action, p_c, p_d))
            soc = self._next_soc(soc, p_c, p_d, dt)

        net_load = [p_c - p_d for _, _, p_c, p_d in plan]
        bus_dict = sced_instance.data["elements"]["bus"][self.bus]
        sced_instance.data["elements"]["load"][self.load_name] = {
            "bus": self.bus,
            "in_service": True,
            "p_load": {"data_type": "time_series", "values": net_load},
            "q_load": {"data_type": "time_series", "values": [0.0] * n_periods},
            "area": bus_dict.get("area"),
            "zone": bus_dict.get("zone"),
        }

        self._pending = {"plan": plan[0], "dt": dt}

    def after_sced(self, options, simulator, sced_instance, lmp_sced):
        """Commit the first SCED period: update SoC and log the step."""
        if self._pending is None:
            return
        da_lmp, action, p_c, p_d = self._pending["plan"]
        dt = self._pending["dt"]
        self._pending = None

        rt_lmp = lmp_sced.data["elements"]["bus"][self.bus]["lmp"]["values"][0]
        revenue = (p_d - p_c) * rt_lmp * dt

        soc_start = self.soc
        self.soc = self._next_soc(self.soc, p_c, p_d, dt)

        self._totals["charged_MWh"] += p_c * dt
        self._totals["discharged_MWh"] += p_d * dt
        self._totals["revenue"] += revenue

        now = simulator.time_manager.current_time.datetime
        row = [
            now.date().isoformat(), now.hour, now.minute, self.bus,
            None if da_lmp is None else round(da_lmp, 4),
            round(rt_lmp, 4), action,
            round(p_c, 4), round(p_d, 4), round(p_c - p_d, 4),
            round(soc_start, 4), round(self.soc, 4), round(revenue, 4),
        ]
        # append each step so a crashed long run keeps what it already did
        with open(self.results_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def finalize(self, options, simulator):
        t = self._totals
        print(f"[battery] charged {t['charged_MWh']:.2f} MWh, discharged "
              f"{t['discharged_MWh']:.2f} MWh, RT revenue ${t['revenue']:.2f}, "
              f"final SoC {self.soc:.2f} MWh")
        print(f"[battery] results written to {self.results_path}")


def register_plugins(context, options, plugin_config):
    battery = RuleBasedBattery(plugin_config)
    context.register_initialization_callback(battery.initialize)
    context.register_before_operations_solve_callback(battery.before_sced)
    context.register_after_operations_callback(battery.after_sced)
    context.register_finalization_callback(battery.finalize)
