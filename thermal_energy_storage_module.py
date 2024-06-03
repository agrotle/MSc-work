# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:05:53 2024

@author: agrot
"""

# Standard Library - built in python imports
from dataclasses import dataclass, field
# Third Party - imports we don't maintain, e.g. numpy, pandas
import pandas as pd
import numpy  as np
import pyomo.environ as pyo
from pyXSteam.XSteam import XSteam
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
# Local Import - imports we are responsible for, e.g. utils
from opti import factories
import opti.IO as IO
import math

################################################################################################################
# Thermal energy storage block
################################################################################################################
@dataclass
class ThermalEnergyStorage():
    parameters: dict

    # Heat flows
    Q_storage_cap:   np.ndarray = field(default_factory = factories.build_data_array, metadata={'unit': 'GJ'})
    Q_storage:       np.ndarray = field(default_factory = factories.build_data_array, metadata={'unit': 'GJ'})
    Q_to_tes:        np.ndarray = field(default_factory = factories.build_data_array, metadata={'unit': 'GJ'})
    Q_from_tes:      np.ndarray = field(default_factory = factories.build_data_array, metadata={'unit': 'GJ'})
    Q_loss:          np.ndarray = field(default_factory = factories.build_data_array, metadata={'unit': 'GJ'})
    # Flows
    M_steam_in:      np.ndarray = field(default_factory = factories.build_data_array, metadata={'unit': 'kg/s'})
    M_steam_out:     np.ndarray = field(default_factory = factories.build_data_array, metadata={'unit': 'kg/s'})
    # Pressures
    Pr_steam_in:       float = field(default = float('nan'), metadata={'unit': 'Pa'})
    Pr_steam_out:      float = field(default = float('nan'), metadata={'unit': 'Pa'})
    # Temperatures
    T_steam_in:        float = field(default = float('nan'), metadata={'unit': 'C'})
    T_steam_out:       float = field(default = float('nan'), metadata={'unit': 'C'})
    # Enthalpies 
    h_steam_in_chrg:   float = field(default = float('nan'), metadata={'unit': 'kJ/kg'})
    h_steam_out_chrg:  float = field(default = float('nan'), metadata={'unit': 'kJ/kg'})
    h_steam_in_dchrg:  float = field(default = float('nan'), metadata={'unit': 'kJ/kg'})
    h_steam_out_dchrg: float = field(default = float('nan'), metadata={'unit': 'kJ/kg'})
    # Steam quality
    x_steam_in:        float = field(default = float('nan'), metadata={'unit': '-'})
    x_steam_out:       float = field(default = float('nan'), metadata={'unit': '-'})
    # Cost elements
    build_cost:        float = field(default_factory = factories.build_data_array, metadata={'unit': 'EUR'})
    op_cost:           float = field(default_factory = factories.build_data_array, metadata={'unit': 'EUR'})
    # Numeric constants
    n_mods:            float = field(default = float('nan'), metadata={'unit': '-'})

    # Self
    name:   str = field(default_factory = str)
    sanity: pd.DataFrame = field(default_factory = factories.build_dataframe)
    
    
    
    # Extract variables from parameters
    def __post_init__(self):
        # Read the parameters file
        IO.read_parameters(self, self.parameters)

    def construct_block(self, hour):
        """Construct a thermal energy storage pyomo block for optimisation."""

        print(f'- CONSTRUCTING THERMAL ENERGY STORAGE ({self.name})')

        # Calculate the enthalpy of each steam flow
        # ---------------------------------------------
        #h_in      = calculate_enthalpy(self.Pr_steam_in,      self.T_steam_in,      self.x_steam_in)
        #h_out     = calculate_enthalpy(self.Pr_steam_out,     self.T_steam_out,     self.x_steam_out)
        
        # Fill in the missing data
        self.Pr_steam_in, self.T_steam_in, self.x_steam_in = \
        calculate_missing_parameters(self.Pr_steam_in, self.T_steam_in, self.x_steam_in)
        self.Pr_steam_out, self.T_steam_out, self.x_steam_out = \
        calculate_missing_parameters(self.Pr_steam_out, self.T_steam_out, self.x_steam_out)
        
        # Create pyomo block
        blk = pyo.Block()

        # Number of modules
        blk.n_mods            = pyo.Param(initialize=3, doc='[-]')  
        #blk.n_mods            = pyo.Var(bounds=(1,5), doc='[-]')

        # Mass flows
        # -------------------------
        blk.M_steam_in        = pyo.Var(hour, bounds=(0, 173.5*blk.n_mods), doc='[kg/s]') #173.5*blk.n_mods
        blk.M_steam_in_state  = pyo.Var(hour, within=pyo.Binary)
        blk.M_steam_out       = pyo.Var(hour, bounds=(0, 173.5*blk.n_mods), doc='[kg/s]') #173.5*blk.n_mods
        
        # Heat levels and flows
        # -------------------------
        blk.Q_storage_cap     = pyo.Param(initialize=1317.91,    doc='[GJ]')
        blk.Q_storage         = pyo.Var(hour, bounds=(0, None),  doc='[GJ]')
        blk.Q_to_tes          = pyo.Var(hour, bounds=(0, None),  doc='[GJ]')
        blk.Q_from_tes        = pyo.Var(hour, bounds=(0, None),  doc='[GJ]')
        blk.Q_loss            = pyo.Var(hour, bounds=(0, None),  doc='[GJ]') 

        # Cost elements
        # ---------------------
        blk.build_cost           = pyo.Var(bounds=(0, None), doc='[EUR]') 
        blk.op_cost              = pyo.Var(bounds=(0, None), doc='[EUR]')

        C_build = 28 # [EUR/kWh]
        # Build cost
        def build_costs_tes_rule(_blk):
            return _blk.build_cost == C_build * _blk.Q_storage_cap * _blk.n_mods * 0.000277 * 10**6   # converting from GJ to kWh
        blk.build_costs_tes = pyo.Constraint(rule=build_costs_tes_rule)

        # Operation cost 
        def operation_costs_tes_rule(_blk):
            return _blk.op_cost == 0.02 * C_build * _blk.Q_storage_cap * _blk.n_mods * 0.000277 * 10**6   # converting from GJ to kWh
        blk.operation_costs_tes = pyo.Constraint(rule=operation_costs_tes_rule)

        U = 0.1939 * 70 # W/m2K
        # Heat loss 
        def heat_loss_rule(_blk, t):
            return _blk.Q_loss[t] == U * (((_blk.Q_storage_cap*1e6/(212*2200*math.pi))**(2/3))*3*math.pi) * (282 - 11) * 3600 / 1e9 * _blk.n_mods 
            # calculating the heat loss U * A * (T_storage - T_amb) - A based on a cylindrical design with h = r
        blk.heat_loss = pyo.Constraint(hour, rule=heat_loss_rule)

        # Storage capacity constraint
        def cap_rule(_blk, t):
            return _blk.Q_storage[t] <= _blk.Q_storage_cap * _blk.n_mods
        blk.cap = pyo.Constraint(hour, rule=cap_rule)

        # Heat transfer constraints
        def heat_to_tes_rule(_blk, t):
            return _blk.Q_to_tes[t] == (_blk.M_steam_in[t]*(60**2)*(2760-650))/(10**6)
        blk.heat_to_tes = pyo.Constraint(hour, rule=heat_to_tes_rule)

        def heat_from_tes_rule(_blk, t):
            return _blk.Q_from_tes[t] == (_blk.M_steam_out[t]*(60**2)*(2774-1000))/(10**6)
        blk.heat_from_tes = pyo.Constraint(hour, rule=heat_from_tes_rule)
        

        # Steam mass balance
        def tes_heat_balance_rule(_blk, t):
            if t == 0:
                return _blk.Q_storage[t] == 0
            else:
                return _blk.Q_storage[t] == _blk.Q_storage[t-1] + _blk.Q_to_tes[t-1] - _blk.Q_from_tes[t-1] - _blk.Q_loss[t-1]
        blk.tes_heat_balance = pyo.Constraint(hour, rule=tes_heat_balance_rule)
        return blk

    
    def consume_block(self, blk):
        """Extract the results from the block."""
        # Extract flow and production
        # ----------------------------------------------------------------------
        index_hours = list(blk.Q_storage.keys())
        for h in index_hours:
            self.Q_storage[h]        = blk.Q_storage[h].value
            self.M_steam_in[h]       = blk.M_steam_in[h].value
            self.M_steam_out[h]      = blk.M_steam_out[h].value
            self.Q_to_tes[h]         = blk.Q_to_tes[h].value
            self.Q_from_tes[h]       = blk.Q_from_tes[h].value
            self.Q_loss[h]           = blk.Q_loss[h].value
        return None
        
    # Save hourly results
    # -----------------------------------------------------
    def save_results(self, filepath):
        # Create output variables
        variables = {self.name + '_Q_storage'        : np.around(self.Q_storage,   decimals = 2),
                     self.name + '_M_steam_in'       : np.around(self.M_steam_in,  decimals = 2),
                     self.name + '_M_steam_out'      : np.around(self.M_steam_out, decimals = 2),
                     self.name + '_Q_to_tes'         : np.around(self.Q_to_tes,    decimals = 2),
                     self.name + '_Q_from_tes'       : np.around(self.Q_from_tes,  decimals = 2),
                     self.name + '_Q_loss'           : np.around(self.Q_loss,      decimals = 2),

                     }
        # Check if file exists, if it does open it. Otherwise create it.
        try:
            df_variables = pd.read_csv(filepath)
            df_variables = pd.concat([df_variables, pd.DataFrame.from_dict(variables)], axis = 1)
        except:
            df_variables = pd.DataFrame.from_dict(variables)

        # Save the results
        df_variables.to_csv(filepath, header=True, index=False)
        return None

    # Save the installed capacities
    # ----------------------------------------------------------
    def save_capacities(self, filename): return None

    # Generate sanity check output
    # ----------------------------------------------------------
    def generate_output(self):

        # Save key values for steam_turbine
        self.sanity.loc[f'THERMAL_STORAGE ({self.name})']  = [float('nan'), '']
        # Key figures
        #self.sanity.loc[self.name + '_utilisation']      = [np.mean(self.P_AC) / np.max(self.P_AC) * 100, '%']
        #self.sanity.loc[self.name + '_production_power'] = [np.sum(self.P_AC) / 1e6, 'TWh']
        # Power
        self.sanity.loc[self.name + '_storage_min']  = [np.min(self.Q_storage),  'MWh']
        self.sanity.loc[self.name + '_storage_mean'] = [np.mean(self.Q_storage), 'MWh']
        self.sanity.loc[self.name + '_storage_max']  = [np.max(self.Q_storage),  'MWh']
        # Steam flows
        self.sanity.loc[self.name + '_steam_in_min']         = [np.min(self.M_steam_in),       'kg_per_h']
        self.sanity.loc[self.name + '_steam_in_mean']        = [np.mean(self.M_steam_in),      'kg_per_h']
        self.sanity.loc[self.name + '_steam_in_max']         = [np.max(self.M_steam_in),       'kg_per_h']
        self.sanity.loc[self.name + '_steam_out_min']        = [np.min(self.M_steam_out),      'kg_per_h']
        self.sanity.loc[self.name + '_steam_out_mean']       = [np.mean(self.M_steam_out),     'kg_per_h']
        self.sanity.loc[self.name + '_steam_out_max']        = [np.max(self.M_steam_out),      'kg_per_h']
        # Heat flows
        self.sanity.loc[self.name + '_heat_in_min']          = [np.min(self.Q_to_tes),       'kJ_per_h']
        self.sanity.loc[self.name + '_heat_in_mean']         = [np.mean(self.Q_to_tes),      'kJ_per_h']
        self.sanity.loc[self.name + '_heat_in_max']          = [np.max(self.Q_to_tes),       'kJ_per_h']
        self.sanity.loc[self.name + '_heat_out_min']         = [np.min(self.Q_from_tes),     'kJ_per_h']
        self.sanity.loc[self.name + '_heat_out_mean']        = [np.mean(self.Q_from_tes),    'kJ_per_h']
        self.sanity.loc[self.name + '_heat_out_max']         = [np.max(self.Q_from_tes),     'kJ_per_h']
        self.sanity.loc[self.name + '_heat_loss_min']        = [np.min(self.Q_loss),         'kJ_per_h']
        self.sanity.loc[self.name + '_heat_loss_mean']       = [np.mean(self.Q_loss),        'kJ_per_h']
        self.sanity.loc[self.name + '_heat_loss_max']        = [np.max(self.Q_loss),         'kJ_per_h']
        # Steam temperatures
        self.sanity.loc[self.name + '_temperature_in']      = [np.min(self.T_steam_in),      'C']
        self.sanity.loc[self.name + '_temperature_out']     = [np.min(self.T_steam_out),     'C']
        # Steam pressures
        self.sanity.loc[self.name + '_pressure_in']      = [np.min(self.Pr_steam_in)      / 1e5, 'bar']
        self.sanity.loc[self.name + '_pressure_out']     = [np.min(self.Pr_steam_out)     / 1e5, 'bar']
        # Steam quality
        self.sanity.loc[self.name + '_quality_in']      = [100 * np.min(self.x_steam_in),      '%']
        self.sanity.loc[self.name + '_quality_out']     = [100 * np.min(self.x_steam_out),     '%']
        

    # -----------------------------------------------------------------------
    # Auxillary functions
    # -----------------------------------------------------------------------

def calculate_missing_parameters(Pr:float, T:float, x:float) -> float:
    """Calculate missing steam parameters from the steam table."""
    # If steam quality is missing
    if np.isnan(x):
        # If steam quality is missing
        if not np.isnan(Pr) and not np.isnan(T):
            T_sat = steamTable.tsat_p(Pr / 1e5)
            # If temperature below saturation temperature
            if T > T_sat:
                return Pr, T, 1
            # If temperature above saturation temperature
            if T < T_sat:
                return Pr, T, 0
        # If temperature and pressure is not given
        return float('nan'), float('nan'), float('nan')
    else:
        # If temperature is given
        if not np.isnan(T):
            Pr = steamTable.psat_t(T) * 1e5
            return Pr, T, x
        # If pressure is given
        if not np.isnan(Pr):
            T = steamTable.tsat_p(Pr / 1e5)
            return Pr, T, x
        # If neither is given
        return float('nan'), float('nan'), float('nan')
            
def calculate_enthalpy(Pr:float, T:float, x:float) -> float:
    """Calculate the enthalpy of steam from the steam table."""
    # Convert units
    Pr = Pr / 1e5
    # If temperature is not given
    if np.isnan(T) and not np.isnan(x):
        return steamTable.h_px(Pr, x) / 1e3
    # If pressure is not given
    elif np.isnan(Pr) and not np.isnan(x):
        return steamTable.h_tx(T, x) / 1e3
    # If steam quality is not given
    elif np.isnan(x):
        # Check that the steam is superheated, otherwise give error
        T_sat = steamTable.tsat_p(Pr)
        if T < T_sat:
            raise ValueError(f'Steam temperature {T} C is below saturation temperature {T_sat} C')
        # Calculate the enthalpy from temperature and pressure (in MJ/kg)
        return steamTable.h_pt(Pr, T) / 1e3
    # If not enough data is given
    else:
        # Return 0 since not enough data is given
        return 0

