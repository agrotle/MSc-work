#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard Library - built in python imports
import time
from pathlib import Path
import os
from datetime import datetime, timedelta
# Third Party - imports we don't maintain, e.g. numpy, pandas
import pandas as pd
import numpy as np
from pyomo.core import Set
import pyomo.environ as pyo
from pyomo.contrib import appsi

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Local Import - imports we are responsible for, e.g. utils
#import Optil.h2heat   as h2heat
import opti.IO as IO
from opti.h2heat import steam_to_heat

import optiNuc.components.steam_turbine_module           as steam_turbine_module
import optiNuc.components.thermal_energy_storage_module  as tes_module                           #added 13.03
import optiNuc.components.direct_air_capture_module      as direct_air_capture_module
import opti.components.LTE_module                        as LTE_module
import opti.components.HTE_module                        as HTE_module
import opti.components.grid_module                       as grid_module
 
import opti.markets.market_spot_module as market_spot_module
import opti.markets.market_gas_module  as market_gas_module
import opti.markets.market_heat_module as market_heat_module


"""
Version 2021/09/23
"""
# Supress warning
pd.set_option('mode.chained_assignment', None)

def run(folder_path, filepath_parameters = 'parameters.csv'):
    ########### INPUT SETTINGS #############################################
    # Get original working directory
    original_cwd = os.getcwd()

    # Create dataframe to key output
    df_sanity = pd.DataFrame(columns = ['value', 'unit'])

    # Change into folder_path
    os.chdir(folder_path)
    print("Optil running from path:", os.getcwd())
    

    ########### CREATE MODEL ################################################

    # Create model to attach components
    mdl   = pyo.ConcreteModel()
    hour = list(range(0,8760))
    mdl.T = Set(initialize=hour)

    # Input files
    print(filepath_parameters)
    df_parameters = pd.read_csv(filepath_parameters, index_col=0)
    filepath_capacities = IO.df_to_value(df_parameters, 'filepath_capacity_input')
    dirpath_output      = IO.df_to_value(df_parameters, 'dirpath_output')
    print(dirpath_output)

    # Create model to attach components
    mdl   = pyo.ConcreteModel()
    mdl.T = Set(initialize=hour)
    
    print(" ")
    print(" Constructing markets")
    print(" -------------------------------------------------------------------------------------")
    tic = time.time()
    # Construct spot market model
    market_spot      = market_spot_module.SpotMarket(filepath_parameters)
    mdl.market_spot = market_spot.construct_block(mdl.T)
    # Construct hydrogen market
    market_H2 = market_gas_module.GasMarket(filepath_parameters)
    mdl.market_H2 = market_H2.construct_block(mdl.T)
    # Construct CO2 market
    market_CO2 = market_gas_module.GasMarket(filepath_parameters)
    mdl.market_CO2 = market_CO2.construct_block(mdl.T)
    # Construct process heat market
    market_heat = market_heat_module.HeatMarket(filepath_parameters)
    mdl.market_heat = market_heat.construct_block(mdl.T)

    toc = time.time()
    model_time_markets = toc-tic
    print(" -------------------------------------------------------------------------------------")
    print(" Construction time: {:1.2f} s".format(model_time_markets))
    print(" ")

    print(" ")
    print(" Constructing components")
    print(" -------------------------------------------------------------------------------------")
    tic = time.time()
    
    # Add turbine blocks
    # ---------------------------
    # High pressure turbine
    HP_steam_turbine = steam_turbine_module.SteamTurbine(filepath_parameters)
    mdl.HP_steam_turbine = HP_steam_turbine.construct_block(mdl.T)
    # Intermidiate pressure turbine
    IP_steam_turbine = steam_turbine_module.SteamTurbine(filepath_parameters)
    mdl.IP_steam_turbine = IP_steam_turbine.construct_block(mdl.T)
    # Low pressure turbine
    LP_steam_turbine = steam_turbine_module.SteamTurbine(filepath_parameters)
    mdl.LP_steam_turbine = LP_steam_turbine.construct_block(mdl.T)

    # Add thermal storage system (added 13.03.24)
    # -----------------------------------------------------
    Thermal_storage = tes_module.ThermalEnergyStorage(filepath_parameters)
    mdl.Thermal_storage = Thermal_storage.construct_block(mdl.T)

    # Add heat consuming blocks
    # -----------------------------------
    # Add DAC block
    direct_air_capture = direct_air_capture_module.DirectAirCapture(filepath_parameters)
    direct_air_capture.read_capacities(filepath_capacities)
    mdl.DAC = direct_air_capture.construct_block(mdl.T)

    # Add electrolyzers
    # ---------------------------------------
    # Add low temperature electrolyzer module
    LTE     = LTE_module.LowTemperatureElectrolyzer(filepath_parameters)
    LTE.read_capacities(filepath_capacities)
    mdl.LTE = LTE.construct_block(mdl.T)
    # Add high temperature electrolyzer module
    HTE     = HTE_module.HighTemperatureElectrolyzer(filepath_parameters)
    HTE.read_capacities(filepath_capacities)
    mdl.HTE = HTE.construct_block(mdl.T)

    # Add infrastructure
    # --------------------------------------
    # Add pipeline block
    # Add grid block
    grid     = grid_module.Grid(filepath_parameters)
    grid.read_capacities(filepath_capacities)
    mdl.grid = grid.construct_block(mdl.T)

    # STEAM FLOWS
    # -------------------------------------------------
    mdl.M_steam_extracted = pyo.Var(hour, bounds=(0, 68.6), doc='[kg/s]')
    mdl.Q_heat_extracted  = pyo.Var(hour, bounds=(0, None), doc='[MW]')
    MW_from_steam = (steam_to_heat(IP_steam_turbine.Pr_steam_in, IP_steam_turbine.T_steam_in, 60))
    
    # Flows extracted to TES (added 15.03)
    mdl.M_steam_tes             = pyo.Var(hour, bounds=(0, 173.5*mdl.Thermal_storage.n_mods), doc='[kg/s]')
    mdl.M_steam_extracted_tes   = pyo.Var(hour, bounds=(0, 173.5*mdl.Thermal_storage.n_mods), doc='[kg/s]')
    
    # Heat extracted in steam
    def steam_to_heat_rule(_mdl, t):
        return _mdl.Q_heat_extracted[t] == _mdl.M_steam_extracted[t] * MW_from_steam
    mdl.steam_to_heat = pyo.Constraint(mdl.T, rule=steam_to_heat_rule)
    # Into high pressure turbine
    def HP_steam_to_turbine_rule(_mdl, t):
        return _mdl.HP_steam_turbine.M_steam_in[t] == 2453.4 - _mdl.M_steam_extracted_tes[t] #changed 11.04
    mdl.HP_steam_to_turbine = pyo.Constraint(mdl.T, rule=HP_steam_to_turbine_rule)
    # Into intermidiate pressure turbine
    def IP_steam_to_turbine_rule(_mdl, t):
        return _mdl.IP_steam_turbine.M_steam_in[t] == (1674.67 - _mdl.M_steam_extracted_tes[t]) - _mdl.M_steam_extracted[t]  + _mdl.M_steam_tes[t]   #changed 24.04
    mdl.IP_steam_to_turbine = pyo.Constraint(mdl.T, rule=IP_steam_to_turbine_rule)
    # Into low pressure turbine
    def LP_steam_to_turbine_rule(_mdl, t):
        return _mdl.LP_steam_turbine.M_steam_in[t] == _mdl.IP_steam_turbine.M_steam_out[t]
    mdl.LP_steam_to_turbine = pyo.Constraint(mdl.T, rule=LP_steam_to_turbine_rule)
    
    # HEAT FLOWS
    # -------------------------------------------------
    # Heat to consumers
    def heat_used_rule(_mdl, t):
        return _mdl.Q_heat_extracted[t] + _mdl.market_heat.Q_heat[t] == _mdl.DAC.Q_heat[t]
    mdl.heat_used = pyo.Constraint(mdl.T, rule=heat_used_rule)
    
    # Steam to thermal storage (added 15.04)
    def heat_to_tes_rule(_mdl, t):
        return _mdl.M_steam_extracted_tes[t] == _mdl.Thermal_storage.M_steam_in[t]
    mdl.heat_to_tes = pyo.Constraint(mdl.T, rule=heat_to_tes_rule)

    # Steam from thermal storage (added 15.04)
    def heat_from_tes_rule(_mdl, t):
        return _mdl.M_steam_tes[t] == _mdl.Thermal_storage.M_steam_out[t]
    mdl.heat_from_tes = pyo.Constraint(mdl.T, rule=heat_from_tes_rule)
    
    # POWER FLOWS
    # -------------------------------------------------
    def power_to_market_rule(_mdl, t):
        return _mdl.market_spot.P_spot[t] == _mdl.HP_steam_turbine.P_AC[t] + _mdl.IP_steam_turbine.P_AC[t] + _mdl.LP_steam_turbine.P_AC[t] - \
                                    _mdl.LTE.P_AC[t] - _mdl.HTE.P_AC[t] - _mdl.DAC.P_AC[t] # MW output of the reactors
    mdl.power_to_market = pyo.Constraint(mdl.T, rule=power_to_market_rule)


    # (added 22.04)
    mdl.MW_produced_turbines = pyo.Var(hour, bounds=(0, None), doc='[MW]')
    def power_produced_rule(_mdl, t):
        return _mdl.MW_produced_turbines[t] == _mdl.HP_steam_turbine.P_AC[t] + _mdl.IP_steam_turbine.P_AC[t] + _mdl.LP_steam_turbine.P_AC[t]
        # MW output of the turbines
    mdl.power_produced = pyo.Constraint(mdl.T, rule=power_produced_rule)
    
    # HYDROGEN FLOWS
    # -------------------------------------------------
    def hydrogen_to_market_rule(_mdl, t):
        return _mdl.market_H2.M_gas[t] + _mdl.LTE.M_H2[t] + _mdl.HTE.M_H2[t] == 0
    mdl.hydrogen_to_market = pyo.Constraint(mdl.T, rule=hydrogen_to_market_rule)

    # CARBON DIOXIDE FLOWS
    # ------------------------------------------------
    def carbon_dioxide_to_market_rule(_mdl, t):
        return _mdl.market_CO2.M_gas[t] + _mdl.DAC.M_CO2[t] == 0
    mdl.carbon_dioxide_to_market = pyo.Constraint(mdl.T, rule=carbon_dioxide_to_market_rule)

    toc = time.time()
    model_time_units = toc-tic
    print(" -------------------------------------------------------------------------------------")
    print(" Construction time: {:1.2f} s".format(model_time_units))
    print(" ")

    # Set power balance in system (power purchased from PPA and spot)
    #def rule_power_balance(model, h):
    #    return mdl.grid.P_AC[h] == model.market_spot.P_spot[h] - model.market_spot.P_dump[h] + model.market_spot.P_slack[h]
    #mdl.power_balance = pyo.Constraint(hour, rule=rule_power_balance)
    
    # Define objective function
    # ---------------------------------
    hoursinyear = 8765.8
    def obj_rule(mdl):
        return len(hour)/hoursinyear * (mdl.LTE.fixed_cost + mdl.HTE.fixed_cost + mdl.grid.fixed_cost -\
                                        mdl.market_spot.total_costs + mdl.market_H2.total_costs + mdl.market_CO2.total_costs + \
                                        mdl.market_heat.total_costs)
    mdl.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    

    # Set objective function
    # ------------------------------------------

    ########### OPIMIZE MODEL ################################################
    solvername = IO.df_to_value(df_parameters, 'solvername')

    print(" ")
    print(" Finding optimal solutions using " + solvername + ' solver')
    print(" -------------------------------------------------------------------------------------")
    
    # Solve model
    # ----------------------------------------------------------------

    solver = pyo.SolverFactory(solvername) #SolverFactory("gurobi", solver_io="python")
    # Check that solver is available
    if not solver.available(): raise RuntimeError('Solver ', solver, ' not available')
    if market_spot.model == 'real': solver.options['NonConvex'] = 2
    #if solvername == "gurobi":      solver.options['mipgap']    = para['solvergap']
    tic = time.time()
    solver.solve(mdl, tee=True) #results = 
    toc = time.time(); model_time_solve = toc - tic
    print(" -------------------------------------------------------------------------------------")
    print(" Solving time: {:1.2f} s".format(model_time_solve))
    print(" ")
    
    # RECALCULATE COMPONENTS BY USING NON-LINEAR EQUATIONS
    # --------------------------------------------------------------
    tic = time.time()

    # Consume the blocks to further analyse the results
    # --------------------------------------------

    print(" ")
    print(" Post-processing solution")
    print(" -------------------------------------------------------------------------------------")
    tic = time.time()
    # Markets
    market_spot.consume_block(mdl.market_spot)
    market_H2.consume_block(mdl.market_H2)
    market_CO2.consume_block(mdl.market_CO2)
    #market_heat.consume_block(mdl.market_heat)
    # Steam turbines
    HP_steam_turbine.consume_block(mdl.HP_steam_turbine)
    IP_steam_turbine.consume_block(mdl.IP_steam_turbine)
    LP_steam_turbine.consume_block(mdl.LP_steam_turbine)
    # Thermal energy storage
    Thermal_storage.consume_block(mdl.Thermal_storage) #added 15.04
    # Electrolyzers
    LTE.consume_block(mdl.LTE)
    HTE.consume_block(mdl.HTE)
    # Direct air capture
    direct_air_capture.consume_block(mdl.DAC)
    # Infrastructure
    grid.consume_block(mdl.grid)

    toc = time.time()
    model_time_postprocessing = toc-tic
    print(" -------------------------------------------------------------------------------------")
    print(" Solving time: {:1.2f} s".format(model_time_postprocessing))
    print(" ")


    print(" ")
    print(" Saving hourly dispatch and installed capacities")
    print(" -------------------------------------------------------------------------------------")

    # Save output to variables (hourly data)
    # -----------------------------------------
    filepath = dirpath_output + "/variables.csv"
    # Steam turbines
    HP_steam_turbine.save_results(filepath)
    IP_steam_turbine.save_results(filepath)
    LP_steam_turbine.save_results(filepath)
    # Thermal energy storage
    Thermal_storage.save_results(filepath) #added 15.04
    # Electrolyzers
    LTE.save_results(filepath)
    HTE.save_results(filepath)
    # Direct air capture
    direct_air_capture.save_results(filepath)
    # Infrastructure
    grid.save_results(filepath)
    # Markets
    market_spot.save_results(dirpath_output)
    market_H2.save_results(dirpath_output)


    # Saved solved capacities (yearly investments)
    # -----------------------------------------
    # Electrolyzers
    LTE.save_capacities(filepath_capacities)
    HTE.save_capacities(filepath_capacities)
    # Direct air capture
    direct_air_capture.save_capacities(filepath_capacities)
    # Grid
    grid.save_capacities(filepath_capacities)

    # Generate sanity check output
    # --------------------------------------------
    # Steam handling
    HP_steam_turbine.generate_output()
    df_sanity = pd.concat([df_sanity, HP_steam_turbine.sanity])
    IP_steam_turbine.generate_output()
    df_sanity = pd.concat([df_sanity, IP_steam_turbine.sanity])
    LP_steam_turbine.generate_output()
    df_sanity = pd.concat([df_sanity, LP_steam_turbine.sanity])
    # Thermal storage
    Thermal_storage.generate_output()                          #added 15.04
    df_sanity = pd.concat([df_sanity, Thermal_storage.sanity]) #added 15.04
    # Markets
    market_spot.generate_sanity_check()
    df_sanity = pd.concat([df_sanity, market_spot.sanity])
    market_H2.generate_sanity_check()
    df_sanity = pd.concat([df_sanity, market_H2.sanity])
    # Electrolyzers
    LTE.generate_sanity_check()
    df_sanity = pd.concat([df_sanity, LTE.sanity])
    HTE.generate_sanity_check()
    df_sanity = pd.concat([df_sanity, HTE.sanity])
    # Direct air capture
    direct_air_capture.generate_sanity_check()
    df_sanity = pd.concat([df_sanity, direct_air_capture.sanity])
    # Infrastructure
    grid.generate_sanity_check()
    df_sanity = pd.concat([df_sanity, grid.sanity])

    # Calculate the power of the turbines
    P_turbines = HP_steam_turbine.P_AC + IP_steam_turbine.P_AC + LP_steam_turbine.P_AC
    print(P_turbines)

    LTE_rel_capacity = np.max(LTE.P_AC) / np.max(P_turbines) * 100
    HTE_rel_capacity = np.max(HTE.P_AC) / np.max(P_turbines) * 100
    df_sanity.loc['LTE_rel_capacity']         = [LTE_rel_capacity, '%']
    df_sanity.loc['HTE_rel_capacity']         = [LTE_rel_capacity, '%']

    # Save the sanity check output
    df_sanity.to_csv(dirpath_output + '/sanity.csv')


    toc = time.time()
    model_time_saving = toc-tic
    print(" -------------------------------------------------------------------------------------")
    print(" Solving time: {:1.2f} s".format(model_time_saving))
    print(" ")
    
    df_sanity.loc['MODEL_TIMING']              = [0,                         '']
    df_sanity.loc['model_time_markets']        = [model_time_markets,       's']
    df_sanity.loc['model_time_units']          = [model_time_units,          's']
    df_sanity.loc['model_time_solving']        = [model_time_solve,          's']
    df_sanity.loc['model_time_postprocessing'] = [model_time_postprocessing, 's']
    df_sanity.loc['model_time_saving']         = [model_time_saving,         's']

    print("")
    print(" ---------------------------------------------------- ")
    print("               FINAL PROJECT SUMMARY                  ")
    print(" ---------------------------------------------------- ")
    print("")
    print(" - Capacities --------------------------------------- ")
    print("Turbine during TES discharge: {:7.1f} MW, {:7.1f} [%] increase from nom.".format(np.max(P_turbines), ((np.max(P_turbines))/1717.01*100)-100))          # added 22.04
    print("Turbine during TES charge:    {:7.1f} MW, {:7.1f} [%] decrease from nom.".format(np.min(P_turbines), (100-(np.min(P_turbines))/1717.01*100)))          # added 22.04
    print(f"Turbine (nominally):          {1717.01:7.1f} MW ")       # added 22.04 - {np.median(P_turbines):7.1f}
    print("DAC:     {:7.1f} MWe, {:7.1f} ton/h ".format(np.max(direct_air_capture.P_AC), np.sum(direct_air_capture.capacity_active) / 1e3))
    print("LTE:     {:7.1f} MWe, {:7.2f} ton/h, ({:1.1f} %)".format(np.max(LTE.P_AC), sum(LTE.capacity_active_stack)/1000, LTE_rel_capacity))
    print("HTE:     {:7.1f} MWe, {:7.2f} ton/h, ({:1.1f} %)".format(np.max(HTE.P_AC), sum(HTE.capacity_active_stack)/1000, HTE_rel_capacity))
    print("")
    print(" - Outputs and inputs ------------------------------- ")
    print("Generation capacity factor:                {:5.2f}  %".format((np.sum(pyo.value(mdl.MW_produced_turbines[h]) for h in hour))/(1717.01*hoursinyear)*100))       
    print("Electricity production:                    {:5.2f}  TWh per year".format(np.sum(market_spot.power) / 1e6))        # changed 22.04,  (for sales)
    print("Electricity price:                         {:5.2f}  EUR per MWh".format(np.mean(market_spot.price)))
    print("Electricity sales:                         {:5.2f}  MEUR".format(np.sum(market_spot.price * market_spot.power / 1e6)))
    print("Hydrogen production:                     {:5.2f}  ktons per year".format(np.abs(np.sum(market_H2.M_gas)) / 1e6))
    print("Hydrogen price:                          {:5.2f}  EUR / kg".format(np.mean(market_H2.price)))
    print("Hydrogen sales:                          {:5.2f}  MEUR".format(- np.sum(market_H2.price * market_H2.M_gas) / 1e6))
    print("Carbon capture:                          {:5.2f}  ktons per year".format(np.abs(np.sum(market_CO2.M_gas)) / 1e6))
    print("Carbon price:                            {:5.2f}  EUR / kg".format(np.mean(market_CO2.price)))
    print("Carbon sales:                            {:5.2f}  MEUR".format(- np.sum(market_CO2.price * market_CO2.M_gas) / 1e6))
    print("Number of TES units:                      {:5.2f} ".format(pyo.value(mdl.Thermal_storage.n_mods)))
    print("TES capacity limit (per unit):             {:5.2f}  GJ".format(pyo.value(mdl.Thermal_storage.Q_storage_cap)))
    print("TES capacity limit utilization:            {:5.2f}  %".format(np.max(Thermal_storage.Q_storage)/(pyo.value(mdl.Thermal_storage.Q_storage_cap)*mdl.Thermal_storage.n_mods)*100))                   # added 15.04, *5
    print("TES total capacity utilization:            {:5.2f}  %".format(np.sum(Thermal_storage.Q_storage)/(pyo.value(mdl.Thermal_storage.Q_storage_cap)*hoursinyear*mdl.Thermal_storage.n_mods)*100))       # added 17.04, *5
    print("TES mean capacity (per unit):              {:5.2f}  GJ".format(np.sum(Thermal_storage.Q_storage) /(mdl.Thermal_storage.n_mods*hoursinyear)))          # added 15.04, *5
    print("TES build cost (per unit):                 {:5.2f}  MEUR".format(pyo.value(mdl.Thermal_storage.build_cost) / (mdl.Thermal_storage.n_mods * 1e6)))
    print("TES annual operation cost (per unit):     {:5.2f}  MEUR".format(pyo.value(mdl.Thermal_storage.op_cost) / (mdl.Thermal_storage.n_mods * 1e6)))
    print("TES charging efficiency:                   {:5.2f} % (assumed constant)".format(97))
    print("TES annual average discharge efficiency:   {:5.2f} %".format(np.sum(Thermal_storage.Q_from_tes)/np.sum(Thermal_storage.Q_to_tes)*100))
    print("TES storage efficiency:                    {:5.2f} %".format(76.45))
    rt_eff = 0.7645*0.97*np.sum(Thermal_storage.Q_from_tes)/np.sum(Thermal_storage.Q_to_tes)
    print("TES round trip efficiency:                 {:5.2f} %".format(rt_eff*100))
    print("TES annual heat/power discharge:           {:7.1f} GJ or {:7.1f} MWh_el".format(np.sum(Thermal_storage.Q_from_tes), np.sum(Thermal_storage.Q_from_tes)*0.2777777778*0.37))
    LCoS = (pyo.value(mdl.Thermal_storage.build_cost) + 20 * pyo.value(mdl.Thermal_storage.op_cost))/(20*1*365*1*pyo.value(mdl.Thermal_storage.n_mods)*135.84*rt_eff)
    print("Est. LCoS using {:5.0f} modules:           {:7.2f} EUR/MWh_el".format(pyo.value(mdl.Thermal_storage.n_mods), pyo.value(LCoS)))
    print("TES heat loss (per unit) and total         {:5.2f}  GJ / {:5.2f}  GJ ".format(np.max(Thermal_storage.Q_loss)/pyo.value(mdl.Thermal_storage.n_mods), np.max(Thermal_storage.Q_loss)))
    print("")
    
    # Go to original path
    os.chdir(original_cwd)