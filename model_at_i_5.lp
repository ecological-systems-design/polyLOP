\ Model plastics_optimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2.665783275098002 flow[agricultural_residue,_from_barley_straw]
   + 1.378639523916458 flow[agricultural_residue,_from_maize_stover]
   + 1.688875579435161 flow[agricultural_residue,_from_rapeseed_straw]
   + 1.721566783626793 flow[agricultural_residue,_from_rice_straw]
   + 3.62372625362852 flow[agricultural_residue,_from_sorghum_straw]
   + 2.943469711678615 flow[agricultural_residue,_from_soybean_straw]
   + 1.639725202115857 flow[agricultural_residue,_from_sugarcane_tops_and_leaves]
   + 3.331575426027551 flow[agricultural_residue,_from_wheat_straw]
   + 4.3050484492671118e-04 flow[benzene,_from_methanol-to-aromatics]
   + 5.89128e-06 flow[cooling_water]
   + 0.00162540963662734 flow[ethylene,_from_methanol-to-olefins]
   + 0.41793855931166 flow[heat,_high_temperature,_from_forest_residue]
   + 9.9577141908369516e-04 flow[heat,_low_temperature,_from_electricity]
   + 0.41793855931166 flow[heat,_low_temperature,_from_forest_residue]
   + 0.00423391167206904 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   + 134.9830189121265 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + 0.1916421856311717 flow[hydrogen,_from_PEM_electrolysis]
   + 2.9475e-10 flow[methanol,_from_agricultural_residue_gasification,_base]
   + 7.7968929530044661e-04 flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   + 0.00293674661526422 flow[methanol,_from_biogenic_co2_hydrogenation,_long_rotation]
   + 9.1429661526421912e-04 flow[methanol,_from_biogenic_co2_hydrogenation,_short_rotation]
   + 14.7372788529408 flow[methanol,_from_forest_residue_gasification,_base]
   + 0.0193986142953004 flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   + 0.00495919661526422 flow[methanol,_from_fossil_co2_hydrogenation]
   + 0.00194836220635025 flow[nitrogen,_from_crogenic_air_separation,_base]
   + 9.1562376910448676e-04 flow[nitrogen,_from_crogenic_air_separation,_improved]
   + 0.00226421691739792 flow[oxygen,_from_cryogenic_air_separation,_base]
   + 8.9433019307880114e-04 flow[oxygen,_from_cryogenic_air_separation,_improved]
   + 0.00162540963662734 flow[propylene,_from_methanol-to-olefins]
   + 4.3050484492671118e-04 flow[p-xylene,_from_methanol-to-aromatics]
   + 2.86754e-05 flow[steam,_high_temperature,_from_heat]
   + 2.86754e-05 flow[steam,_low_temperature,_from_heat]
   + 4.3050484492671118e-04 flow[toluene,_from_methanol-to-aromatics]
   + 0.00212935760256857 flow[ammonia,_from_Haber-Bosch_process]
   + 0.00441989927271127 flow[ammonium_sulfate,_from_ammonia_and_sulfuric_acid]
   + 0.00154131307282826 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   + 22.75862216293076 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   - 0.00136845387603339 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   + 15.26122680372089 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   + 0.2099288060317943 flow[corn_steep_liquor,_from_corn]
   + 0.00651804529675101 flow[sulfur_dioxide,_from_sulfur]
   + 0.00194656318920879 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 33.21136500248186 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 0.00209470602847552 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 29.88952073192093 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.00446550585545285 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 34.05648921712942 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.00304464238884748 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 32.04859955605484 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0049580673812151 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 36.57611168566222 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.00178333122147348 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 13.34386122820447 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.00178333122147348 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 13.34386122820447 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.00178333122147348 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 13.34386122820447 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.00124386412573887 flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + 11.54134401095256 flow[xylose,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   - 8.4137337420119506e-04 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + 22.50934219107386 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 11.46362276016234 flow[xylose_mother_liquor,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 0.00131805019485119 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 11.70287992978419 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 0.00131805019485119 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 11.70287992978419 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 8.9076138589457758e-04 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   + 0.00448405514265955 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   + 0.0351648884759928 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   + 0.00135356807532409 flow[HDPE,_from_gas_phase_process]
   + 9.9367186931862491e-04 flow[HDPE,_from_slurry_process]
   + 0.00345823109916887 flow[LDPE,_from_autoclave_process]
   + 0.00294966577266274 flow[LDPE,_from_tubular_process]
   + 0.00131446121073658 flow[LLDPE,_from_gas_phase_process]
   + 0.00136137754750984 flow[LLDPE,_from_slurry_process]
   + 0.00128017905405923 flow[LLDPE,_from_solution_process]
   + 6.4116768365239749e-04 flow[1-hexene,_from_ethylene_trimerization]
   + 0.0495623022757119 flow[potato_starch,_from_potato]
   + 0.0223451237884209 flow[enzyme,_from_potato_starch]
   + 1.9271636303245596e-04 flow[PLA,_from_ring-opening_polymerization]
   + 0.0414668249104957 flow[PHB,_from_glucose_bs_fermentation]
   + 0.0719886395432776 flow[PHB,_from_glucose_bl_fermentation]
   + 0.0367764836809599 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   + 0.0367764836809599 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   + 2.4842512863300034e-04 flow[PBS,_from_1,4-BDO_and_succinic_acid]
   + 0.0391527297698227 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 0.0620027297698227 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   + 3.951301016855989e-04 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   + 0.00352865290403314 flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   + 0.000733984 flow[syngas,_from_agricultural_residue_gasification]
   + 13.66457058398731 flow[syngas,_from_forest_residue_gasification]
   + 0.00795156165020711 flow[hydrogen,_from_partial_condensation_of_syngas]
   + 0.000663083 flow[carbon_monoxide,_from_partial_condensation_of_syngas]
   + 4.0288125156197417e-04 flow[acetic_acid,_from_methanol_carbonylation]
   + 0.0075642552762262 flow[sodium_acetate,_from_sodium_hydroxide_and_acetic_acid]
   + 0.00426409499537493 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   + 0.00529310263092517 flow[vinyl_chloride,_from_ethylene_chlorination]
   + 0.00369094645243624 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   + 7.5894216075281598e-04 flow[PVC,_from_bulk_polymerization]
   + 0.00193434724024567 flow[PP,_from_loop_reactor_process]
   + 2.3005473615915831e-04 flow[terephthalic_acid,_from_p-xylene_oxidation]
   + 5.88419e-05 flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   + 2.32462e-05 flow[ethylbenzene,_from_liquid_phase_alkylation]
   + 0.00252718353025076 flow[hydrogen_peroxide,_from_anthraquinone_process]
   + 0.00289263893274988 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   + 0.00289263893274988 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   + 0.00289263893274988 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   + 0.0106429054314416 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   + 0.0106429054314416 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   + 0.0106429054314416 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   + 3.4730655368893452e-04 flow[ethylbenzene,_from_butadiene]
   + 2.4226630945555145e-04 flow[styrene,_from_ethylbenzene]
   + 2.0251987565127137e-04 flow[styrene,_from_benzene_and_ethylene]
   + 0.00248720261441047 flow[styrene,_from_toluene_and_methanol]
   + 1.6492043042093605e-04 flow[GPPS,_from_continuous_bulk_polymerization]
   + 9.9829478466552358e-04 flow[polybutadiene,_from_solution_polymerization]
   + 2.2188022720564548e-04 flow[HIPS,_from_continuous_bulk_polymerization]
Subject To
 supply_flow_barley_straw: flow[agricultural_residue,_from_barley_straw]
   <= 53.19647815127136
 supply_flow_co2_feedstock_biogenic_long:
   - 14.11253 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + 1.45732 flow[methanol,_from_biogenic_co2_hydrogenation,_long_rotation]
   - 0.8608695652173913 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.8608695652173913 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.8608695652173913 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.8608695652173913 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.8608695652173913 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.86087 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   - 0.88 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - 0.9209302325581395 flow[PHB,_from_glucose_bl_fermentation] <= 1e+16
 supply_flow_co2_feedstock_biogenic_short:
   - 14.11253 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   + 1.45732 flow[methanol,_from_biogenic_co2_hydrogenation,_short_rotation]
   - 0.8608695652173913 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.8608695652173913 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.8608695652173913 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.8608695652173913 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.8608695652173913 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.86087 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 0.88 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - 0.9209302325581395 flow[PHB,_from_glucose_bs_fermentation]
   + 0.26 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   + 0.26 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   + 0.1788 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 0.1788 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   <= 1e+16
 supply_flow_co2_feedstock_fossil:
   1.45732 flow[methanol,_from_fossil_co2_hydrogenation] <= 1e+16
 supply_flow_cooling_water_kg: 64.25958 flow[HDPE,_from_gas_phase_process]
   + 70.9359 flow[HDPE,_from_slurry_process]
   + 122.67739 flow[LDPE,_from_autoclave_process]
   + 85.95762999999999 flow[LDPE,_from_tubular_process]
   + 87.62671 flow[LLDPE,_from_gas_phase_process]
   + 82.53601 flow[LLDPE,_from_slurry_process]
   + 92.63395 flow[LLDPE,_from_solution_process]
   + 181.09519 flow[1-hexene,_from_ethylene_trimerization]
   + 95 flow[PLA,_from_ring-opening_polymerization]
   + 712.5 flow[PHB,_from_glucose_bs_fermentation]
   + 712.5 flow[PHB,_from_glucose_bl_fermentation]
   + 0.26 flow[PBS,_from_1,4-BDO_and_succinic_acid]
   + 213.4 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 213.4 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   + 275.39822 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   + 232.83667 flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   + 166.3085678297872 flow[hydrogen,_from_partial_condensation_of_syngas]
   + 13.868520592 flow[carbon_monoxide,_from_partial_condensation_of_syngas]
   + 67.93156 flow[acetic_acid,_from_methanol_carbonylation]
   + 1093.24747 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   + 143.12362 flow[vinyl_chloride,_from_ethylene_chlorination]
   + 244.52024 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   + 36.21904 flow[PVC,_from_bulk_polymerization]
   + 33.04779 flow[PP,_from_loop_reactor_process]
   + 130.18825 flow[terephthalic_acid,_from_p-xylene_oxidation]
   + 6.10049 flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   + 6.75977 flow[ethylbenzene,_from_liquid_phase_alkylation]
   + 260.3765 flow[hydrogen_peroxide,_from_anthraquinone_process]
   + 82.61947000000001 flow[ethylbenzene,_from_butadiene]
   + 163.56985 flow[styrene,_from_ethylbenzene]
   + 179.67647 flow[styrene,_from_benzene_and_ethylene]
   + 774.45317 flow[styrene,_from_toluene_and_methanol]
   + 16.84102 flow[GPPS,_from_continuous_bulk_polymerization]
   + 162.30135 flow[polybutadiene,_from_solution_polymerization]
   + 11.65018 flow[HIPS,_from_continuous_bulk_polymerization] <= 1e+16
 supply_flow_electricity:
   0.1213055555555555 flow[benzene,_from_methanol-to-aromatics]
   + 0.00166 flow[cooling_water]
   + 0.458 flow[ethylene,_from_methanol-to-olefins]
   + 0.2805836139169472 flow[heat,_low_temperature,_from_electricity]
   + 1.193011111111111 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   + 1.193011111111111 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + 54 flow[hydrogen,_from_PEM_electrolysis]
   + 0.219697 flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   + 0.257626 flow[methanol,_from_biogenic_co2_hydrogenation,_long_rotation]
   + 0.257626 flow[methanol,_from_biogenic_co2_hydrogenation,_short_rotation]
   + 0.219697 flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   + 0.257626 flow[methanol,_from_fossil_co2_hydrogenation]
   + 0.549 flow[nitrogen,_from_crogenic_air_separation,_base]
   + 0.258 flow[nitrogen,_from_crogenic_air_separation,_improved]
   + 0.638 flow[oxygen,_from_cryogenic_air_separation,_base]
   + 0.252 flow[oxygen,_from_cryogenic_air_separation,_improved]
   + 0.458 flow[propylene,_from_methanol-to-olefins]
   + 0.1213055555555555 flow[p-xylene,_from_methanol-to-aromatics]
   + 0.00808 flow[steam,_high_temperature,_from_heat]
   + 0.00808 flow[steam,_low_temperature,_from_heat]
   + 0.1213055555555555 flow[toluene,_from_methanol-to-aromatics]
   + 0.6 flow[ammonia,_from_Haber-Bosch_process]
   + 0.1971 flow[ammonium_sulfate,_from_ammonia_and_sulfuric_acid]
   - 0.18 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 0.18 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   - 0.3901576236799667 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 0.3901576236799667 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   + 6.8474358974358971e-04 flow[corn_steep_liquor,_from_corn]
   + 0.416 flow[sulfur_dioxide,_from_sulfur]
   - 0.5755971786906208 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.5755971786906208 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 0.0698915733990414 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 0.0698915733990414 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 1.258268898651719 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 1.258268898651719 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.8579048766849858 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.8579048766849858 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 1.397060534041661 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 1.397060534041661 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.5172236463179214 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.5172236463179214 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.5172236463179214 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.5172236463179214 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.5172236463179214 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.5172236463179214 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.4470464830759329 flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 0.4470464830759329 flow[xylose,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   - 0.3728093396990591 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 0.3728093396990591 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 0.3713934777270102 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 0.3713934777270102 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 0.3713934777270102 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 0.3713934777270102 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 0.2307692307692308 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   + 0.3 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   + 0.3 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   + 0.3814 flow[HDPE,_from_gas_phase_process]
   + 0.27999 flow[HDPE,_from_slurry_process]
   + 0.97444 flow[LDPE,_from_autoclave_process]
   + 0.83114 flow[LDPE,_from_tubular_process]
   + 0.37038 flow[LLDPE,_from_gas_phase_process]
   + 0.3836 flow[LLDPE,_from_slurry_process]
   + 0.36072 flow[LLDPE,_from_solution_process]
   + 0.18066 flow[1-hexene,_from_ethylene_trimerization]
   + 0.1675 flow[potato_starch,_from_potato]
   + 6.2963 flow[enzyme,_from_potato_starch]
   + 0.0543 flow[PLA,_from_ring-opening_polymerization]
   + 1.392921775898524 flow[PHB,_from_glucose_bs_fermentation]
   + 1.392921775898524 flow[PHB,_from_glucose_bl_fermentation]
   + 1.76 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   + 1.76 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   + 0.07 flow[PBS,_from_1,4-BDO_and_succinic_acid]
   + 1.3832 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 1.3832 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   + 0.11133 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   + 0.99428 flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   + 0.206818375 flow[syngas,_from_agricultural_residue_gasification]
   + 0.206818375 flow[syngas,_from_forest_residue_gasification]
   + 2.240547574468085 flow[hydrogen,_from_partial_condensation_of_syngas]
   + 0.186839924 flow[carbon_monoxide,_from_partial_condensation_of_syngas]
   + 0.11352 flow[acetic_acid,_from_methanol_carbonylation]
   + 0.333 flow[sodium_acetate,_from_sodium_hydroxide_and_acetic_acid]
   + 1.16717 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   + 0.06949 flow[vinyl_chloride,_from_ethylene_chlorination]
   + 1.04001 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   + 0.21385 flow[PVC,_from_bulk_polymerization]
   + 0.54505 flow[PP,_from_loop_reactor_process]
   + 0.06482 flow[terephthalic_acid,_from_p-xylene_oxidation]
   + 0.01658 flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   + 0.00655 flow[ethylbenzene,_from_liquid_phase_alkylation]
   + 0.71209 flow[hydrogen_peroxide,_from_anthraquinone_process]
   + 0.8150731660044578 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   + 0.8150731660044578 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   + 0.8150731660044578 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   + 0.8150731660044578 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   + 0.8150731660044578 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   + 0.8150731660044578 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   + 0.09786 flow[ethylbenzene,_from_butadiene]
   + 0.06826 flow[styrene,_from_ethylbenzene]
   + 0.05706 flow[styrene,_from_benzene_and_ethylene]
   + 0.70081 flow[styrene,_from_toluene_and_methanol]
   + 0.04647 flow[GPPS,_from_continuous_bulk_polymerization]
   + 0.28129 flow[polybutadiene,_from_solution_polymerization]
   + 0.06252 flow[HIPS,_from_continuous_bulk_polymerization] <= 1e+16
 supply_flow_forest_residue:
   0.0555555555555556 flow[heat,_high_temperature,_from_forest_residue]
   + 0.0555555555555556 flow[heat,_low_temperature,_from_forest_residue]
   + 17.9893 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + 1.96356 flow[methanol,_from_forest_residue_gasification,_base]
   + 3.03 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   + 2.02881964313583 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   + 4.423664932583077 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 3.98208056577627 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 4.536857578397212 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 4.269764820412973 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 4.871850336159018 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 1.779941612076332 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 1.779941612076332 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 1.779941612076332 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 1.533567170907273 flow[xylose,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 3.000457508296839 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 1.533567170907273 flow[xylose_mother_liquor,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 1.56101 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 1.56101 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 1.82086956521739 flow[syngas,_from_forest_residue_gasification]
   <= 952.77106341868
 supply_flow_lime:
   0.13 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   + 0.13 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   + 0.3476483703153201 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 0.3476483703153201 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 0.069336264156353 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 0.069336264156353 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 0.0720384 flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + 0.0720384 flow[xylose,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   <= 1e+16
 supply_flow_maize_grain:
   0.0503076923076923 flow[corn_steep_liquor,_from_corn] <= 1e+16
 supply_flow_maize_stover: flow[agricultural_residue,_from_maize_stover]
   <= 458.6468514535616
 supply_flow_rapeseed_straw:
   flow[agricultural_residue,_from_rapeseed_straw] <= 36.71128183683346
 supply_flow_rice_straw: flow[agricultural_residue,_from_rice_straw]
   <= 413.6657483989756
 supply_flow_sodium_hydroxide:
   0.001301 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   + 0.001301 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   + 0.16254379445801 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 0.16254379445801 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 0.0870884503273283 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 0.0870884503273283 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 0.00576923076923077 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   + 0.06 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   + 0.06 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   + 0.2 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   + 0.2 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   + 0.2204 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 0.2204 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   + 0.513 flow[sodium_acetate,_from_sodium_hydroxide_and_acetic_acid]
   <= 1e+16
 supply_flow_sorghum_straw: flow[agricultural_residue,_from_sorghum_straw]
   <= 97.54963715382405
 supply_flow_soybean_straw: flow[agricultural_residue,_from_soybean_straw]
   <= 120.4458629322265
 supply_flow_sugarcane_tops_and_leaves:
   flow[agricultural_residue,_from_sugarcane_tops_and_leaves]
   <= 192.0928689001732
 supply_flow_sulfur: 0.55 flow[sulfur_dioxide,_from_sulfur] <= 1e+16
 supply_flow_sulfuric_acid:
   0.76039 flow[ammonium_sulfate,_from_ammonia_and_sulfuric_acid]
   + 0.41 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   + 0.41 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   + 0.3068797112219981 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 0.3068797112219981 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 0.1370000477851579 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 0.1370000477851579 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 0.0106807177442324 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 0.0106807177442324 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 0.0106807177442324 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 0.0106807177442324 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 0.0106807177442324 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 0.0106807177442324 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 0.050319627601164 flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + 0.050319627601164 flow[xylose,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 0.0984514453066253 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + 0.0984514453066253 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 0.02489 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   <= 1e+16
 supply_flow_water: 0.61127 flow[cooling_water]
   + 9.51 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   + 9.51 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + 14 flow[hydrogen,_from_PEM_electrolysis]
   + 2.9475 flow[methanol,_from_agricultural_residue_gasification,_base]
   + 1.666146 flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   + 2.9475 flow[methanol,_from_forest_residue_gasification,_base]
   + 1.666146 flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   + 0.0618182 flow[steam,_high_temperature,_from_heat]
   + 0.0618182 flow[steam,_low_temperature,_from_heat]
   + 18.25 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   + 18.25 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   + 16.3967 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   + 16.3967 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   + 0.0869206246153846 flow[corn_steep_liquor,_from_corn]
   + 14.36336129100754 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 14.36336129100754 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 9.141921919051942 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 9.141921919051942 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 16.3966681184669 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 16.3966681184669 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 14.8298918891223 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 14.8298918891223 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 18.20555393159895 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + 18.20555393159895 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + 7.691896895471376 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 7.691896895471376 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 7.691896895471376 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 7.691896895471376 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 7.691896895471376 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + 7.691896895471376 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + 14.8298918891223 flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + 14.8298918891223 flow[xylose,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 14.8298918891223 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + 14.8298918891223 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + 9.859472033990759 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 9.859472033990759 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 9.859472033990759 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 9.859472033990759 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 0.0865384615384615 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   + 2.93 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   + 2.93 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   + 0.5049 flow[HDPE,_from_slurry_process]
   + 1.27685 flow[LDPE,_from_autoclave_process]
   + 0.47235 flow[LDPE,_from_tubular_process]
   + 3.727 flow[potato_starch,_from_potato]
   + 2.631578947368421 flow[PHB,_from_glucose_bs_fermentation]
   + 2.631578947368421 flow[PHB,_from_glucose_bl_fermentation]
   + 3.49 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   + 3.49 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   + 0.49 flow[PBS,_from_1,4-BDO_and_succinic_acid]
   + 24.6813 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 24.6813 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   + 2.58707 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   + 1.58563 flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   + 1.2343 flow[syngas,_from_agricultural_residue_gasification]
   + 1.2343 flow[syngas,_from_forest_residue_gasification]
   + 10.3483 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   + 1.1767 flow[terephthalic_acid,_from_p-xylene_oxidation]
   + 0.13436 flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   + 4.00579 flow[hydrogen_peroxide,_from_anthraquinone_process]
   + 18.6161449752883 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   + 18.6161449752883 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   + 18.6161449752883 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   + 18.6161449752883 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   + 18.6161449752883 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   + 18.6161449752883 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   + 0.66763 flow[ethylbenzene,_from_butadiene]
   + 0.35885 flow[styrene,_from_ethylbenzene]
   + 2.21153 flow[polybutadiene,_from_solution_polymerization] <= 1e+16
 supply_flow_wheat_straw: flow[agricultural_residue,_from_wheat_straw]
   <= 254.1575917525405
 supply_flow_yeast:
   0.02 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   + 0.02 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   + 0.09 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   + 0.09 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   <= 1e+16
 supply_flow_refrigerant_energy:
   1.107692307692308 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   <= 1e+16
 supply_flow_phosphate_rock:
   0.08 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   + 0.08 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   <= 1e+16
 supply_flow_potato: 3.92 flow[potato_starch,_from_potato] <= 1e+16
 supply_flow_SDS:
   0.2857142857142857 flow[PHB,_from_glucose_bs_fermentation]
   + 0.2857142857142857 flow[PHB,_from_glucose_bl_fermentation] <= 1e+16
 supply_flow_sodium_hypochlorite:
   0.1375 flow[PHB,_from_glucose_bs_fermentation]
   + 0.1375 flow[PHB,_from_glucose_bl_fermentation] <= 1e+16
 supply_flow_potassium_sulfate:
   0.0186 flow[PHB,_from_glucose_bs_fermentation]
   + 0.0186 flow[PHB,_from_glucose_bl_fermentation] <= 1e+16
 supply_flow_magnesium_sulfate:
   0.0209 flow[PHB,_from_glucose_bs_fermentation]
   + 0.0209 flow[PHB,_from_glucose_bl_fermentation] <= 1e+16
 supply_flow_hydrochloric_acid:
   0.402 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   + 0.402 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   + 0.15648 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 0.15648 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   <= 1e+16
 supply_flow_magnesium_oxide:
   0.2380952380952381 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   + 0.2380952380952381 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   + 0.1637619047619047 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 0.1637619047619047 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   <= 1e+16
 supply_flow_sodium_phosphate:
   0.1826 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   + 0.1826 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   <= 1e+16
 supply_flow_chlorine:
   0.5787 flow[vinyl_chloride,_from_ethylene_chlorination] <= 1e+16
 demand_flow_HDPE: flow[HDPE,_from_gas_phase_process]
   + flow[HDPE,_from_slurry_process] >= 70
 demand_flow_LDPE: flow[LDPE,_from_autoclave_process]
   + flow[LDPE,_from_tubular_process] >= 40
 demand_flow_LLDPE: flow[LLDPE,_from_gas_phase_process]
   + flow[LLDPE,_from_slurry_process] + flow[LLDPE,_from_solution_process]
   >= 40
 demand_flow_PLA: flow[PLA,_from_ring-opening_polymerization] >= 15
 demand_flow_PHB: flow[PHB,_from_glucose_bs_fermentation]
   + flow[PHB,_from_glucose_bl_fermentation] >= 15
 demand_flow_PBS: flow[PBS,_from_1,4-BDO_and_succinic_acid]
   + flow[PBS,_from_PBS_bs] + flow[PBS,_from_PBS_bl] >= 15
 demand_flow_CA:
   flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride] >= 15
 demand_flow_PVC: flow[PVC,_from_bulk_polymerization] >= 40
 demand_flow_PP: flow[PP,_from_loop_reactor_process] >= 80
 demand_flow_PET: flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   >= 40
 demand_flow_GPPS: flow[GPPS,_from_continuous_bulk_polymerization] >= 15
 demand_flow_HIPS: flow[HIPS,_from_continuous_bulk_polymerization] >= 15
 intermediate_flow_agricultural_residue:
   flow[agricultural_residue,_from_barley_straw]
   + flow[agricultural_residue,_from_maize_stover]
   + flow[agricultural_residue,_from_rapeseed_straw]
   + flow[agricultural_residue,_from_rice_straw]
   + flow[agricultural_residue,_from_sorghum_straw]
   + flow[agricultural_residue,_from_soybean_straw]
   + flow[agricultural_residue,_from_sugarcane_tops_and_leaves]
   + flow[agricultural_residue,_from_wheat_straw]
   - 0.0634920634920635 flow[heat,_high_temperature,_from_agricultural_residue]
   - 0.0634920634920635 flow[heat,_low_temperature,_from_agricultural_residue]
   - 17.9893 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   - 1.96356 flow[methanol,_from_agricultural_residue_gasification,_base]
   - 1.10995 flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   - 1.10995 flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   - 3.03 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 2.02881964313583 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 4.423664932583077 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 3.98208056577627 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 4.536857578397212 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 4.269764820412973 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 4.871850336159018 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 1.779941612076332 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.779941612076332 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.779941612076332 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.533567170907273 flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 3.000457508296839 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 1.533567170907273 flow[xylose_mother_liquor,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 1.56101 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 1.56101 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 1.82086956521739 flow[syngas,_from_agricultural_residue_gasification]
   >= 0
 intermediate_flow_agricultural_residue_upper_bound:
   flow[agricultural_residue,_from_barley_straw]
   + flow[agricultural_residue,_from_maize_stover]
   + flow[agricultural_residue,_from_rapeseed_straw]
   + flow[agricultural_residue,_from_rice_straw]
   + flow[agricultural_residue,_from_sorghum_straw]
   + flow[agricultural_residue,_from_soybean_straw]
   + flow[agricultural_residue,_from_sugarcane_tops_and_leaves]
   + flow[agricultural_residue,_from_wheat_straw]
   - 0.0634920634920635 flow[heat,_high_temperature,_from_agricultural_residue]
   - 0.0634920634920635 flow[heat,_low_temperature,_from_agricultural_residue]
   - 17.9893 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   - 1.96356 flow[methanol,_from_agricultural_residue_gasification,_base]
   - 1.10995 flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   - 1.10995 flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   - 3.03 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 2.02881964313583 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 4.423664932583077 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 3.98208056577627 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 4.536857578397212 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 4.269764820412973 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 4.871850336159018 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 1.779941612076332 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.779941612076332 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.779941612076332 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.533567170907273 flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 3.000457508296839 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 1.533567170907273 flow[xylose_mother_liquor,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 1.56101 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 1.56101 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 1.82086956521739 flow[syngas,_from_agricultural_residue_gasification]
   <= 0.01
 intermediate_flow_ammonia: flow[ammonia,_from_Haber-Bosch_process]
   - 0.2654 flow[ammonium_sulfate,_from_ammonia_and_sulfuric_acid]
   - 0.288 flow[trimethylamine,_from_methanol_and_ammonia,_mass_balance_only]
   - 0.01 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 0.01 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   - 0.1377003928230173 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.1377003928230173 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.073015721316959 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.073015721316959 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0476190476190476 flow[PHB,_from_glucose_bs_fermentation]
   - 0.0476190476190476 flow[PHB,_from_glucose_bl_fermentation] >= 0
 intermediate_flow_ammonia_upper_bound:
   flow[ammonia,_from_Haber-Bosch_process]
   - 0.2654 flow[ammonium_sulfate,_from_ammonia_and_sulfuric_acid]
   - 0.288 flow[trimethylamine,_from_methanol_and_ammonia,_mass_balance_only]
   - 0.01 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 0.01 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   - 0.1377003928230173 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.1377003928230173 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.073015721316959 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.073015721316959 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0476190476190476 flow[PHB,_from_glucose_bs_fermentation]
   - 0.0476190476190476 flow[PHB,_from_glucose_bl_fermentation] <= 0.01
 intermediate_flow_ammonium_sulfate:
   flow[ammonium_sulfate,_from_ammonia_and_sulfuric_acid]
   - 0.1053687162215086 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 0.1053687162215086 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   - 0.0148 flow[PHB,_from_glucose_bs_fermentation]
   - 0.0148 flow[PHB,_from_glucose_bl_fermentation]
   - 0.108 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   - 0.108 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022] >= 0
 intermediate_flow_ammonium_sulfate_upper_bound:
   flow[ammonium_sulfate,_from_ammonia_and_sulfuric_acid]
   - 0.1053687162215086 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 0.1053687162215086 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   - 0.0148 flow[PHB,_from_glucose_bs_fermentation]
   - 0.0148 flow[PHB,_from_glucose_bl_fermentation]
   - 0.108 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   - 0.108 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022] <= 0.01
 intermediate_flow_benzene: flow[benzene,_from_methanol-to-aromatics]
   - 0.73906 flow[ethylbenzene,_from_liquid_phase_alkylation]
   + 0.15633 flow[styrene,_from_ethylbenzene]
   - 0.78015 flow[styrene,_from_benzene_and_ethylene]
   + 0.0059 flow[styrene,_from_toluene_and_methanol] >= 0
 intermediate_flow_benzene_upper_bound:
   flow[benzene,_from_methanol-to-aromatics]
   - 0.73906 flow[ethylbenzene,_from_liquid_phase_alkylation]
   + 0.15633 flow[styrene,_from_ethylbenzene]
   - 0.78015 flow[styrene,_from_benzene_and_ethylene]
   + 0.0059 flow[styrene,_from_toluene_and_methanol] <= 0.01
 intermediate_flow_cooling_water:
   - 0.4882948615178236 flow[benzene,_from_methanol-to-aromatics]
   + flow[cooling_water]
   - 0.4882948615178236 flow[p-xylene,_from_methanol-to-aromatics]
   - 0.4882948615178236 flow[toluene,_from_methanol-to-aromatics]
   - 2.38256255527233 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 2.38256255527233 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   - 1.776559384790658 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.6631153018381899 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.6631153018381899 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 0.6631153018381899 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.6631153018381899 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 2.076923076923077 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   >= 0
 intermediate_flow_cooling_water_upper_bound:
   - 0.4882948615178236 flow[benzene,_from_methanol-to-aromatics]
   + flow[cooling_water]
   - 0.4882948615178236 flow[p-xylene,_from_methanol-to-aromatics]
   - 0.4882948615178236 flow[toluene,_from_methanol-to-aromatics]
   - 2.38256255527233 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 2.38256255527233 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   - 1.776559384790658 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 1.776559384790658 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.6631153018381899 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.6631153018381899 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 0.6631153018381899 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.6631153018381899 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 2.076923076923077 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   <= 0.01
 intermediate_flow_corn_steep_liquor:
   - 0.0473235187015554 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 0.0473235187015554 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   + flow[corn_steep_liquor,_from_corn]
   - 0.0623739250451216 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0623739250451216 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0561953457256176 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0561953457256176 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0642966027874564 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0642966027874564 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0602039247835221 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0602039247835221 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0684010523238819 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0684010523238819 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.104436249706274 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 0.104436249706274 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   >= 0
 intermediate_flow_corn_steep_liquor_upper_bound:
   - 0.0473235187015554 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 0.0473235187015554 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   + flow[corn_steep_liquor,_from_corn]
   - 0.0623739250451216 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0623739250451216 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0561953457256176 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0561953457256176 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0642966027874564 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0642966027874564 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0602039247835221 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0602039247835221 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0684010523238819 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.0684010523238819 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.104436249706274 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - 0.104436249706274 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   <= 0.01
 intermediate_flow_ethanol_bl:
   flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   - flow[ethanol,_from_ethanol_bl] >= 0
 intermediate_flow_ethanol_bl_upper_bound:
   flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   + flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   - flow[ethanol,_from_ethanol_bl] <= 0.01
 intermediate_flow_ethanol_bs:
   flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - flow[ethanol,_from_ethanol_bs] >= 0
 intermediate_flow_ethanol_bs_upper_bound:
   flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   + flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   - flow[ethanol,_from_ethanol_bs] <= 0.01
 intermediate_flow_ethylene: flow[ethylene,_from_methanol-to-olefins]
   + flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   - 1.02 flow[HDPE,_from_gas_phase_process]
   - 1.035 flow[HDPE,_from_slurry_process]
   - 1.009 flow[LDPE,_from_autoclave_process]
   - 1.007 flow[LDPE,_from_tubular_process]
   - 0.915 flow[LLDPE,_from_gas_phase_process]
   - 0.915 flow[LLDPE,_from_slurry_process]
   - 0.92099 flow[LLDPE,_from_solution_process]
   - 1.09195 flow[1-hexene,_from_ethylene_trimerization]
   - 0.68 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   - 0.4674 flow[vinyl_chloride,_from_ethylene_chlorination]
   - 0.59847 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   - 0.26517 flow[ethylbenzene,_from_liquid_phase_alkylation]
   + flow[ethylene,_from_ethylene_bs] + flow[ethylene,_from_ethylene_bl]
   - 0.2805 flow[styrene,_from_benzene_and_ethylene] >= 0
 intermediate_flow_ethylene_upper_bound:
   flow[ethylene,_from_methanol-to-olefins]
   + flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   - 1.02 flow[HDPE,_from_gas_phase_process]
   - 1.035 flow[HDPE,_from_slurry_process]
   - 1.009 flow[LDPE,_from_autoclave_process]
   - 1.007 flow[LDPE,_from_tubular_process]
   - 0.915 flow[LLDPE,_from_gas_phase_process]
   - 0.915 flow[LLDPE,_from_slurry_process]
   - 0.92099 flow[LLDPE,_from_solution_process]
   - 1.09195 flow[1-hexene,_from_ethylene_trimerization]
   - 0.68 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   - 0.4674 flow[vinyl_chloride,_from_ethylene_chlorination]
   - 0.59847 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   - 0.26517 flow[ethylbenzene,_from_liquid_phase_alkylation]
   + flow[ethylene,_from_ethylene_bs] + flow[ethylene,_from_ethylene_bl]
   - 0.2805 flow[styrene,_from_benzene_and_ethylene] <= 0.01
 intermediate_flow_furfural_bl:
   flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   >= 0
 intermediate_flow_furfural_bl_upper_bound:
   flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   <= 0.01
 intermediate_flow_furfural_bs:
   flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   >= 0
 intermediate_flow_furfural_bs_upper_bound:
   flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   <= 0.01
 intermediate_flow_glucose: - 0.774 flow[cellulose,_estimated_from_glucose]
   + flow[glucose,_from_glucose_bl] + flow[glucose,_from_glucose_bs] >= 0
 intermediate_flow_glucose_upper_bound:
   - 0.774 flow[cellulose,_estimated_from_glucose]
   + flow[glucose,_from_glucose_bl] + flow[glucose,_from_glucose_bs]
   <= 0.01
 intermediate_flow_glucose_bl:
   flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.2471036265371662 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 2.77 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - 2.855670103092784 flow[PHB,_from_glucose_bl_fermentation]
   - 1.28 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   - flow[glucose,_from_glucose_bl] >= 0
 intermediate_flow_glucose_bl_upper_bound:
   flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.2471036265371662 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   + flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 2.77 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - 2.855670103092784 flow[PHB,_from_glucose_bl_fermentation]
   - 1.28 flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   - flow[glucose,_from_glucose_bl] <= 0.01
 intermediate_flow_glucose_bs:
   flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.2471036265371662 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 2.77 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - 2.855670103092784 flow[PHB,_from_glucose_bs_fermentation]
   - 1.28 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   - flow[glucose,_from_glucose_bs] >= 0
 intermediate_flow_glucose_bs_upper_bound:
   flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.2471036265371662 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 2.77 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - 2.855670103092784 flow[PHB,_from_glucose_bs_fermentation]
   - 1.28 flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   - flow[glucose,_from_glucose_bs] <= 0.01
 intermediate_flow_heat_high:
   flow[heat,_high_temperature,_from_agricultural_residue]
   + flow[heat,_high_temperature,_from_forest_residue]
   - 1.28 flow[steam,_high_temperature,_from_heat]
   + 12.13838179636009 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 12.13838179636009 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 12.13838179636009 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 12.13838179636009 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 1.730769230769231 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   - 2 flow[sodium_acetate,_from_sodium_hydroxide_and_acetic_acid]
   - 2.658618 flow[vinyl_chloride,_from_ethylene_chlorination]
   - 1.134621 flow[terephthalic_acid,_from_p-xylene_oxidation]
   - 2.219004 flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   - 2.009664 flow[ethylbenzene,_from_liquid_phase_alkylation]
   - 11.59998740188003 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - 11.59998740188003 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - 11.59998740188003 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - 11.59998740188003 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - 11.59998740188003 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - 11.59998740188003 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   - 3.30129 flow[ethylbenzene,_from_butadiene]
   + 2.19807 flow[styrene,_from_ethylbenzene]
   - 2.583252 flow[styrene,_from_benzene_and_ethylene]
   + 6.07086 flow[styrene,_from_toluene_and_methanol]
   - 0.527535 flow[GPPS,_from_continuous_bulk_polymerization]
   - 0.527535 flow[HIPS,_from_continuous_bulk_polymerization] >= 0
 intermediate_flow_heat_high_upper_bound:
   flow[heat,_high_temperature,_from_agricultural_residue]
   + flow[heat,_high_temperature,_from_forest_residue]
   - 1.28 flow[steam,_high_temperature,_from_heat]
   + 12.13838179636009 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 12.13838179636009 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 12.13838179636009 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   + 12.13838179636009 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 1.730769230769231 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   - 2 flow[sodium_acetate,_from_sodium_hydroxide_and_acetic_acid]
   - 2.658618 flow[vinyl_chloride,_from_ethylene_chlorination]
   - 1.134621 flow[terephthalic_acid,_from_p-xylene_oxidation]
   - 2.219004 flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   - 2.009664 flow[ethylbenzene,_from_liquid_phase_alkylation]
   - 11.59998740188003 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - 11.59998740188003 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - 11.59998740188003 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - 11.59998740188003 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - 11.59998740188003 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - 11.59998740188003 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   - 3.30129 flow[ethylbenzene,_from_butadiene]
   + 2.19807 flow[styrene,_from_ethylbenzene]
   - 2.583252 flow[styrene,_from_benzene_and_ethylene]
   + 6.07086 flow[styrene,_from_toluene_and_methanol]
   - 0.527535 flow[GPPS,_from_continuous_bulk_polymerization]
   - 0.527535 flow[HIPS,_from_continuous_bulk_polymerization] <= 0.01
 intermediate_flow_heat_low:
   flow[heat,_low_temperature,_from_agricultural_residue]
   + flow[heat,_low_temperature,_from_electricity]
   + flow[heat,_low_temperature,_from_forest_residue]
   - 1.28 flow[steam,_low_temperature,_from_heat]
   - 0.0356178461538462 flow[corn_steep_liquor,_from_corn]
   - 0.6885 flow[LLDPE,_from_solution_process]
   - 1.624 flow[potato_starch,_from_potato]
   - 8.2667 flow[enzyme,_from_potato_starch] >= 0
 intermediate_flow_heat_low_upper_bound:
   flow[heat,_low_temperature,_from_agricultural_residue]
   + flow[heat,_low_temperature,_from_electricity]
   + flow[heat,_low_temperature,_from_forest_residue]
   - 1.28 flow[steam,_low_temperature,_from_heat]
   - 0.0356178461538462 flow[corn_steep_liquor,_from_corn]
   - 0.6885 flow[LLDPE,_from_solution_process]
   - 1.624 flow[potato_starch,_from_potato]
   - 8.2667 flow[enzyme,_from_potato_starch] <= 0.01
 intermediate_flow_hydrogen:
   flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   + flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + flow[hydrogen,_from_PEM_electrolysis]
   - 0.083991 flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   - 0.193204 flow[methanol,_from_biogenic_co2_hydrogenation,_long_rotation]
   - 0.193204 flow[methanol,_from_biogenic_co2_hydrogenation,_short_rotation]
   - 0.083991 flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   - 0.193204 flow[methanol,_from_fossil_co2_hydrogenation]
   - 0.181041078 flow[ammonia,_from_Haber-Bosch_process]
   + flow[hydrogen,_from_partial_condensation_of_syngas]
   - 0.0627458961375 flow[hydrogen_peroxide,_from_anthraquinone_process]
   + 0.0284911328617114 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   + 0.0284911328617114 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   + 0.0284911328617114 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   + 0.0284911328617114 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   + 0.0284911328617114 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   + 0.0284911328617114 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   + 0.017973505600875 flow[ethylbenzene,_from_butadiene] >= 0
 intermediate_flow_hydrogen_upper_bound:
   flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   + flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + flow[hydrogen,_from_PEM_electrolysis]
   - 0.083991 flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   - 0.193204 flow[methanol,_from_biogenic_co2_hydrogenation,_long_rotation]
   - 0.193204 flow[methanol,_from_biogenic_co2_hydrogenation,_short_rotation]
   - 0.083991 flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   - 0.193204 flow[methanol,_from_fossil_co2_hydrogenation]
   - 0.181041078 flow[ammonia,_from_Haber-Bosch_process]
   + flow[hydrogen,_from_partial_condensation_of_syngas]
   - 0.0627458961375 flow[hydrogen_peroxide,_from_anthraquinone_process]
   + 0.0284911328617114 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   + 0.0284911328617114 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   + 0.0284911328617114 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   + 0.0284911328617114 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   + 0.0284911328617114 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   + 0.0284911328617114 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   + 0.017973505600875 flow[ethylbenzene,_from_butadiene] <= 0.01
 intermediate_flow_lactic_acid_bl:
   flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   - flow[lactic_acid,_from_lactic_acid_bl] >= 0
 intermediate_flow_lactic_acid_bl_upper_bound:
   flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   - flow[lactic_acid,_from_lactic_acid_bl] <= 0.01
 intermediate_flow_lactic_acid_bs:
   flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - flow[lactic_acid,_from_lactic_acid_bs] >= 0
 intermediate_flow_lactic_acid_bs_upper_bound:
   flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - flow[lactic_acid,_from_lactic_acid_bs] <= 0.01
 intermediate_flow_organosolv_lignin_bs:
   flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   >= 0
 intermediate_flow_organosolv_lignin_bs_upper_bound:
   flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   + flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   <= 0.01
 intermediate_flow_organosolv_lignin_bl:
   flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   >= 0
 intermediate_flow_organosolv_lignin_bl_upper_bound:
   flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   + flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   <= 0.01
 intermediate_flow_methanol:
   - 2.591093117 flow[benzene,_from_methanol-to-aromatics]
   - 2.571 flow[ethylene,_from_methanol-to-olefins]
   + flow[methanol,_from_methanol_bl] + flow[methanol,_from_methanol_bs]
   + flow[methanol,_from_methanol_f]
   - 2.571 flow[propylene,_from_methanol-to-olefins]
   - 2.591093117 flow[p-xylene,_from_methanol-to-aromatics]
   - 2.591093117 flow[toluene,_from_methanol-to-aromatics]
   - 1.627 flow[trimethylamine,_from_methanol_and_ammonia,_mass_balance_only]
   - 0.52532 flow[acetic_acid,_from_methanol_carbonylation]
   - 0.58272 flow[styrene,_from_toluene_and_methanol] >= 0
 intermediate_flow_methanol_upper_bound:
   - 2.591093117 flow[benzene,_from_methanol-to-aromatics]
   - 2.571 flow[ethylene,_from_methanol-to-olefins]
   + flow[methanol,_from_methanol_bl] + flow[methanol,_from_methanol_bs]
   + flow[methanol,_from_methanol_f]
   - 2.571 flow[propylene,_from_methanol-to-olefins]
   - 2.591093117 flow[p-xylene,_from_methanol-to-aromatics]
   - 2.591093117 flow[toluene,_from_methanol-to-aromatics]
   - 1.627 flow[trimethylamine,_from_methanol_and_ammonia,_mass_balance_only]
   - 0.52532 flow[acetic_acid,_from_methanol_carbonylation]
   - 0.58272 flow[styrene,_from_toluene_and_methanol] <= 0.01
 intermediate_flow_methanol_bl:
   flow[methanol,_from_biogenic_co2_hydrogenation,_long_rotation]
   + flow[methanol,_from_forest_residue_gasification,_base]
   + flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   - flow[methanol,_from_methanol_bl] >= 0
 intermediate_flow_methanol_bl_upper_bound:
   flow[methanol,_from_biogenic_co2_hydrogenation,_long_rotation]
   + flow[methanol,_from_forest_residue_gasification,_base]
   + flow[methanol,_from_forest_residue_gasification,_combined_with_co2_hydrogenation]
   - flow[methanol,_from_methanol_bl] <= 0.01
 intermediate_flow_methanol_bs:
   flow[methanol,_from_agricultural_residue_gasification,_base]
   + flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   + flow[methanol,_from_biogenic_co2_hydrogenation,_short_rotation]
   - flow[methanol,_from_methanol_bs] >= 0
 intermediate_flow_methanol_bs_upper_bound:
   flow[methanol,_from_agricultural_residue_gasification,_base]
   + flow[methanol,_from_agricultural_residue_gasification,_combined_with_co2_hydrogenation]
   + flow[methanol,_from_biogenic_co2_hydrogenation,_short_rotation]
   - flow[methanol,_from_methanol_bs] <= 0.01
 intermediate_flow_methanol_f:
   flow[methanol,_from_fossil_co2_hydrogenation]
   - flow[methanol,_from_methanol_f] >= 0
 intermediate_flow_methanol_f_upper_bound:
   flow[methanol,_from_fossil_co2_hydrogenation]
   - flow[methanol,_from_methanol_f] <= 0.01
 intermediate_flow_nitrogen_liquid:
   flow[nitrogen,_from_crogenic_air_separation,_base]
   + flow[nitrogen,_from_crogenic_air_separation,_improved]
   - 0.839181773 flow[ammonia,_from_Haber-Bosch_process]
   - 0.06881 flow[HDPE,_from_gas_phase_process]
   - 0.06881 flow[HDPE,_from_slurry_process]
   - 0.00111 flow[LDPE,_from_autoclave_process]
   - 0.00331 flow[LDPE,_from_tubular_process]
   - 0.05601 flow[LLDPE,_from_gas_phase_process]
   - 0.0344 flow[LLDPE,_from_slurry_process]
   - 0.0344 flow[LLDPE,_from_solution_process] >= 0
 intermediate_flow_nitrogen_liquid_upper_bound:
   flow[nitrogen,_from_crogenic_air_separation,_base]
   + flow[nitrogen,_from_crogenic_air_separation,_improved]
   - 0.839181773 flow[ammonia,_from_Haber-Bosch_process]
   - 0.06881 flow[HDPE,_from_gas_phase_process]
   - 0.06881 flow[HDPE,_from_slurry_process]
   - 0.00111 flow[LDPE,_from_autoclave_process]
   - 0.00331 flow[LDPE,_from_tubular_process]
   - 0.05601 flow[LLDPE,_from_gas_phase_process]
   - 0.0344 flow[LLDPE,_from_slurry_process]
   - 0.0344 flow[LLDPE,_from_solution_process] <= 0.01
 intermediate_flow_oxygen_liquid:
   - 4.6624 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   - 4.6624 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + flow[oxygen,_from_cryogenic_air_separation,_base]
   + flow[oxygen,_from_cryogenic_air_separation,_improved]
   - 0.4 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   - 0.8389 flow[syngas,_from_agricultural_residue_gasification]
   - 0.8389 flow[syngas,_from_forest_residue_gasification]
   - 0.1378 flow[vinyl_chloride,_from_ethylene_chlorination]
   - 0.5917 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   - 0.1165 flow[styrene,_from_ethylbenzene] >= 0
 intermediate_flow_oxygen_liquid_upper_bound:
   - 4.6624 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   - 4.6624 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   + flow[oxygen,_from_cryogenic_air_separation,_base]
   + flow[oxygen,_from_cryogenic_air_separation,_improved]
   - 0.4 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   - 0.8389 flow[syngas,_from_agricultural_residue_gasification]
   - 0.8389 flow[syngas,_from_forest_residue_gasification]
   - 0.1378 flow[vinyl_chloride,_from_ethylene_chlorination]
   - 0.5917 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   - 0.1165 flow[styrene,_from_ethylbenzene] <= 0.01
 intermediate_flow_p-xylene: flow[p-xylene,_from_methanol-to-aromatics]
   - 0.66545 flow[terephthalic_acid,_from_p-xylene_oxidation] >= 0
 intermediate_flow_p-xylene_upper_bound:
   flow[p-xylene,_from_methanol-to-aromatics]
   - 0.66545 flow[terephthalic_acid,_from_p-xylene_oxidation] <= 0.01
 intermediate_flow_steam_high:
   - 4.9386 flow[benzene,_from_methanol-to-aromatics]
   - 5.5872 flow[ethylene,_from_methanol-to-olefins]
   - 2.981705454545455 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   - 2.981705454545455 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   - 5.5872 flow[propylene,_from_methanol-to-olefins]
   - 4.9386 flow[p-xylene,_from_methanol-to-aromatics]
   + flow[steam,_high_temperature,_from_heat]
   - 4.9386 flow[toluene,_from_methanol-to-aromatics]
   + 0.2872727272727273 flow[ammonia,_from_Haber-Bosch_process]
   - 1.94439 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 1.94439 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 1.94439 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 1.94439 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 1.6225 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - 1.6225 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - 0.4125 flow[LDPE,_from_autoclave_process]
   - 0.95975 flow[LDPE,_from_tubular_process]
   - 10.89 flow[PHB,_from_glucose_bs_fermentation]
   - 10.89 flow[PHB,_from_glucose_bl_fermentation]
   - 8.0025 flow[PBS,_from_1,4-BDO_and_succinic_acid]
   - 11.9042 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   - 11.9042 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   - 8.1675 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   - 4.7685 flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   + 5.2947027664137 flow[syngas,_from_agricultural_residue_gasification]
   + 5.2947027664137 flow[syngas,_from_forest_residue_gasification]
   - 1.1195 flow[acetic_acid,_from_methanol_carbonylation]
   - 0.814 flow[vinyl_chloride,_from_ethylene_chlorination]
   - 20.9 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   - 0.17215 flow[PVC,_from_bulk_polymerization]
   - 6.4306 flow[terephthalic_acid,_from_p-xylene_oxidation]
   - 0.42075 flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   + 3.08 flow[ethylbenzene,_from_liquid_phase_alkylation]
   - 13.5025 flow[hydrogen_peroxide,_from_anthraquinone_process]
   - 22.22332590367283 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - 22.22332590367283 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - 22.22332590367283 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - 22.22332590367283 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - 22.22332590367283 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - 22.22332590367283 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   - 6.2865 flow[ethylbenzene,_from_butadiene]
   - 5.225 flow[styrene,_from_ethylbenzene]
   - 4.5925 flow[styrene,_from_benzene_and_ethylene]
   - 21.01 flow[styrene,_from_toluene_and_methanol]
   - 11.0275 flow[polybutadiene,_from_solution_polymerization] >= 0
 intermediate_flow_steam_high_upper_bound:
   - 4.9386 flow[benzene,_from_methanol-to-aromatics]
   - 5.5872 flow[ethylene,_from_methanol-to-olefins]
   - 2.981705454545455 flow[hydrogen,_from_agricultural_residue_gasification,_ccu]
   - 2.981705454545455 flow[hydrogen,_from_forest_residue_gasification,_ccu]
   - 5.5872 flow[propylene,_from_methanol-to-olefins]
   - 4.9386 flow[p-xylene,_from_methanol-to-aromatics]
   + flow[steam,_high_temperature,_from_heat]
   - 4.9386 flow[toluene,_from_methanol-to-aromatics]
   + 0.2872727272727273 flow[ammonia,_from_Haber-Bosch_process]
   - 1.94439 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 1.94439 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 1.94439 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 1.94439 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 1.6225 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - 1.6225 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - 0.4125 flow[LDPE,_from_autoclave_process]
   - 0.95975 flow[LDPE,_from_tubular_process]
   - 10.89 flow[PHB,_from_glucose_bs_fermentation]
   - 10.89 flow[PHB,_from_glucose_bl_fermentation]
   - 8.0025 flow[PBS,_from_1,4-BDO_and_succinic_acid]
   - 11.9042 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   - 11.9042 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   - 8.1675 flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   - 4.7685 flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   + 5.2947027664137 flow[syngas,_from_agricultural_residue_gasification]
   + 5.2947027664137 flow[syngas,_from_forest_residue_gasification]
   - 1.1195 flow[acetic_acid,_from_methanol_carbonylation]
   - 0.814 flow[vinyl_chloride,_from_ethylene_chlorination]
   - 20.9 flow[ethylene_glycol,_from_ethylene_and_oxygen]
   - 0.17215 flow[PVC,_from_bulk_polymerization]
   - 6.4306 flow[terephthalic_acid,_from_p-xylene_oxidation]
   - 0.42075 flow[PET,_from_ethylene_glycol_and_terephthalic_acid]
   + 3.08 flow[ethylbenzene,_from_liquid_phase_alkylation]
   - 13.5025 flow[hydrogen_peroxide,_from_anthraquinone_process]
   - 22.22332590367283 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - 22.22332590367283 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - 22.22332590367283 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - 22.22332590367283 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - 22.22332590367283 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - 22.22332590367283 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   - 6.2865 flow[ethylbenzene,_from_butadiene]
   - 5.225 flow[styrene,_from_ethylbenzene]
   - 4.5925 flow[styrene,_from_benzene_and_ethylene]
   - 21.01 flow[styrene,_from_toluene_and_methanol]
   - 11.0275 flow[polybutadiene,_from_solution_polymerization] <= 0.01
 intermediate_flow_steam_low: flow[steam,_low_temperature,_from_heat]
   - 0.0801053830817431 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.76427 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.76427 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 0.76427 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.76427 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 5.746153846153847 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   - 0.11 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - 0.11 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - 0.957 flow[HDPE,_from_slurry_process]
   - 0.0825 flow[LLDPE,_from_gas_phase_process]
   - 0.9625 flow[LLDPE,_from_slurry_process]
   - 1.43 flow[LLDPE,_from_solution_process]
   - 7.92 flow[1-hexene,_from_ethylene_trimerization]
   - 6.03075 flow[PLA,_from_ring-opening_polymerization]
   - 0.2475 flow[PBS,_from_1,4-BDO_and_succinic_acid]
   - 0.38665 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   - 0.38665 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   - 70.95 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   - 0.55275 flow[PP,_from_loop_reactor_process] >= 0
 intermediate_flow_steam_low_upper_bound:
   flow[steam,_low_temperature,_from_heat]
   - 0.0801053830817431 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[furfural,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[furfural,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.0801053830817431 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.76427 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.76427 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   - 0.76427 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.76427 flow[organosolv_lignin,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + 5.746153846153847 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   - 0.11 flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - 0.11 flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - 0.957 flow[HDPE,_from_slurry_process]
   - 0.0825 flow[LLDPE,_from_gas_phase_process]
   - 0.9625 flow[LLDPE,_from_slurry_process]
   - 1.43 flow[LLDPE,_from_solution_process]
   - 7.92 flow[1-hexene,_from_ethylene_trimerization]
   - 6.03075 flow[PLA,_from_ring-opening_polymerization]
   - 0.2475 flow[PBS,_from_1,4-BDO_and_succinic_acid]
   - 0.38665 flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   - 0.38665 flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   - 70.95 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   - 0.55275 flow[PP,_from_loop_reactor_process] <= 0.01
 intermediate_flow_succinic_acid_bl:
   flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   + flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   - flow[succinic_acid,_from_succinic_acid_bl] >= 0
 intermediate_flow_succinic_acid_bl_upper_bound:
   flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   + flow[succinic_acid,_from_glucose_bl,_according_to_Leiva2023]
   - flow[succinic_acid,_from_succinic_acid_bl] <= 0.01
 intermediate_flow_succinic_acid_bs:
   flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   + flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   - flow[succinic_acid,_from_succinic_acid_bs] >= 0
 intermediate_flow_succinic_acid_bs_upper_bound:
   flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   + flow[succinic_acid,_from_glucose_bs,_according_to_Leiva2023]
   - flow[succinic_acid,_from_succinic_acid_bs] <= 0.01
 intermediate_flow_sulfur_dioxide:
   - 0.0189668626124954 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 0.0189668626124954 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   - 0.00704307692307692 flow[corn_steep_liquor,_from_corn]
   + flow[sulfur_dioxide,_from_sulfur] >= 0
 intermediate_flow_sulfur_dioxide_upper_bound:
   - 0.0189668626124954 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 0.0189668626124954 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   - 0.00704307692307692 flow[corn_steep_liquor,_from_corn]
   + flow[sulfur_dioxide,_from_sulfur] <= 0.01
 intermediate_flow_toluene: flow[toluene,_from_methanol-to-aromatics]
   + 0.01892 flow[styrene,_from_ethylbenzene]
   - 1.19892 flow[styrene,_from_toluene_and_methanol] >= 0
 intermediate_flow_toluene_upper_bound:
   flow[toluene,_from_methanol-to-aromatics]
   + 0.01892 flow[styrene,_from_ethylbenzene]
   - 1.19892 flow[styrene,_from_toluene_and_methanol] <= 0.01
 intermediate_flow_trimethylamine:
   flow[trimethylamine,_from_methanol_and_ammonia,_mass_balance_only]
   - 0.04 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 0.04 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   - 0.023784 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 0.023784 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   >= 0
 intermediate_flow_trimethylamine_upper_bound:
   flow[trimethylamine,_from_methanol_and_ammonia,_mass_balance_only]
   - 0.04 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 0.04 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   - 0.023784 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Mandegari2018]
   - 0.023784 flow[lactic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Mandegari2018]
   <= 0.01
 intermediate_flow_xylose_bl:
   flow[xylose,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   >= 0
 intermediate_flow_xylose_bl_upper_bound:
   flow[xylose,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   <= 0.01
 intermediate_flow_xylose_bs:
   flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   >= 0
 intermediate_flow_xylose_bs_upper_bound:
   flow[xylose,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   <= 0.01
 intermediate_flow_xylose_mother_liquor:
   flow[xylose_mother_liquor,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + flow[xylose_mother_liquor,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   >= 0
 intermediate_flow_xylose_mother_liquor_upper_bound:
   flow[xylose_mother_liquor,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_LiuF2021]
   + flow[xylose_mother_liquor,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_LiuF2021]
   <= 0.01
 intermediate_flow_ethylene_bs:
   flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - flow[ethylene,_from_ethylene_bs] >= 0
 intermediate_flow_ethylene_bs_upper_bound:
   flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - flow[ethylene,_from_ethylene_bs] <= 0.01
 intermediate_flow_ethylene_bl:
   flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - flow[ethylene,_from_ethylene_bl] >= 0
 intermediate_flow_ethylene_bl_upper_bound:
   flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - flow[ethylene,_from_ethylene_bl] <= 0.01
 intermediate_flow_BDO_14_bs:
   flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - flow[1,4-BDO,_from_BDO_14_bs] >= 0
 intermediate_flow_BDO_14_bs_upper_bound:
   flow[1,4-BDO,_from_glucose_bs_fermentation,_according_to_Leiva2023]
   - flow[1,4-BDO,_from_BDO_14_bs] <= 0.01
 intermediate_flow_BDO_14_bl:
   flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - flow[1,4-BDO,_from_BDO_14_bl] >= 0
 intermediate_flow_BDO_14_bl_upper_bound:
   flow[1,4-BDO,_from_glucose_bl_fermentation,_according_to_Leiva2023]
   - flow[1,4-BDO,_from_BDO_14_bl] <= 0.01
 intermediate_flow_hexene_1: - 0.095 flow[LLDPE,_from_gas_phase_process]
   - 0.095 flow[LLDPE,_from_slurry_process]
   - 0.0896 flow[LLDPE,_from_solution_process]
   + flow[1-hexene,_from_ethylene_trimerization] >= 0
 intermediate_flow_hexene_1_upper_bound:
   - 0.095 flow[LLDPE,_from_gas_phase_process]
   - 0.095 flow[LLDPE,_from_slurry_process]
   - 0.0896 flow[LLDPE,_from_solution_process]
   + flow[1-hexene,_from_ethylene_trimerization] <= 0.01
 intermediate_flow_enzyme:
   - 0.02 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 0.02 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   - 0.2109034929398025 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.2109034929398025 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.1751326038132556 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.1751326038132556 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.1983885017421603 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.1983885017421603 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.218271250704514 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.218271250704514 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.2215726395790704 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.2215726395790704 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0278551532033426 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.0278551532033426 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.00525982979128037 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.00525982979128037 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + flow[enzyme,_from_potato_starch] >= 0
 intermediate_flow_enzyme_upper_bound:
   - 0.02 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Kosamia2023]
   - 0.02 flow[succinic_acid,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Kosamia2023]
   - 0.2109034929398025 flow[ethanol,_from_dilute_acid_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.2109034929398025 flow[ethanol,_from_dilute_acid_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.1751326038132556 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.1751326038132556 flow[ethanol,_from_steam_explosion_with_acid_catalyst_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.1983885017421603 flow[ethanol,_from_steam_explosion_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.1983885017421603 flow[ethanol,_from_steam_explosion_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.218271250704514 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.218271250704514 flow[ethanol,_from_liquid_hot_water_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.2215726395790704 flow[ethanol,_from_wet_oxidation_pre-treatment_of_agricultural_residue,_according_to_Wang2013]
   - 0.2215726395790704 flow[ethanol,_from_wet_oxidation_pre-treatment_of_forest_residue,_according_to_Wang2013]
   - 0.0278551532033426 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Moncada2017]
   - 0.0278551532033426 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Moncada2017]
   - 0.00525982979128037 flow[glucose,_from_organosolv_pre-treatment_of_agricultural_residue,_according_to_Nitzsche2016]
   - 0.00525982979128037 flow[glucose,_from_organosolv_pre-treatment_of_forest_residue,_according_to_Nitzsche2016]
   + flow[enzyme,_from_potato_starch] <= 0.01
 intermediate_flow_potato_starch: flow[potato_starch,_from_potato]
   - 4.1667 flow[enzyme,_from_potato_starch] >= 0
 intermediate_flow_potato_starch_upper_bound:
   flow[potato_starch,_from_potato]
   - 4.1667 flow[enzyme,_from_potato_starch] <= 0.01
 intermediate_flow_propylene: flow[propylene,_from_methanol-to-olefins]
   - 1.045 flow[PP,_from_loop_reactor_process]
   + flow[propylene,_from_propylene_bs]
   + flow[propylene,_from_propylene_bl] >= 0
 intermediate_flow_propylene_upper_bound:
   flow[propylene,_from_methanol-to-olefins]
   - 1.045 flow[PP,_from_loop_reactor_process]
   + flow[propylene,_from_propylene_bs]
   + flow[propylene,_from_propylene_bl] <= 0.01
 intermediate_flow_ethanol:
   - 1.677912 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   + flow[ethanol,_from_ethanol_bl] + flow[ethanol,_from_ethanol_bs]
   - 1.907135381335401 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - 1.907135381335401 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - 1.907135381335401 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - 1.907135381335401 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - 1.907135381335401 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - 1.907135381335401 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   >= 0
 intermediate_flow_ethanol_upper_bound:
   - 1.677912 flow[ethylene,_from_ethanol_dehydration,_according_to_Nitzsche2016]
   + flow[ethanol,_from_ethanol_bl] + flow[ethanol,_from_ethanol_bs]
   - 1.907135381335401 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - 1.907135381335401 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - 1.907135381335401 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - 1.907135381335401 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - 1.907135381335401 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - 1.907135381335401 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   <= 0.01
 intermediate_flow_lactic_acid: flow[lactic_acid,_from_lactic_acid_bs]
   + flow[lactic_acid,_from_lactic_acid_bl]
   - 1.3288 flow[PLA,_from_ring-opening_polymerization] >= 0
 intermediate_flow_lactic_acid_upper_bound:
   flow[lactic_acid,_from_lactic_acid_bs]
   + flow[lactic_acid,_from_lactic_acid_bl]
   - 1.3288 flow[PLA,_from_ring-opening_polymerization] <= 0.01
 intermediate_flow_BDO_14: flow[1,4-BDO,_from_BDO_14_bs]
   + flow[1,4-BDO,_from_BDO_14_bl]
   - 0.68 flow[PBS,_from_1,4-BDO_and_succinic_acid] >= 0
 intermediate_flow_BDO_14_upper_bound: flow[1,4-BDO,_from_BDO_14_bs]
   + flow[1,4-BDO,_from_BDO_14_bl]
   - 0.68 flow[PBS,_from_1,4-BDO_and_succinic_acid] <= 0.01
 intermediate_flow_succinic_acid:
   flow[succinic_acid,_from_succinic_acid_bs]
   + flow[succinic_acid,_from_succinic_acid_bl]
   - 0.69 flow[PBS,_from_1,4-BDO_and_succinic_acid] >= 0
 intermediate_flow_succinic_acid_upper_bound:
   flow[succinic_acid,_from_succinic_acid_bs]
   + flow[succinic_acid,_from_succinic_acid_bl]
   - 0.69 flow[PBS,_from_1,4-BDO_and_succinic_acid] <= 0.01
 intermediate_flow_PBS_bs:
   flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   - flow[PBS,_from_PBS_bs] >= 0
 intermediate_flow_PBS_bs_upper_bound:
   flow[PBS,_from_glucose_bs,_according_to_Ioannidou2022]
   - flow[PBS,_from_PBS_bs] <= 0.01
 intermediate_flow_PBS_bl:
   flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   - flow[PBS,_from_PBS_bl] >= 0
 intermediate_flow_PBS_bl_upper_bound:
   flow[PBS,_from_glucose_bl,_according_to_Ioannidou2022]
   - flow[PBS,_from_PBS_bl] <= 0.01
 intermediate_flow_cellulose: flow[cellulose,_estimated_from_glucose]
   - 0.70107 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   >= 0
 intermediate_flow_cellulose_upper_bound:
   flow[cellulose,_estimated_from_glucose]
   - 0.70107 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   <= 0.01
 intermediate_flow_acetaldehyde:
   flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   - 1.0798 flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   >= 0
 intermediate_flow_acetaldehyde_upper_bound:
   flow[acetaldehyde,_from_ethylene_by_one-step_oxidation]
   - 1.0798 flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   <= 0.01
 intermediate_flow_acetic_anhydride:
   flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   - 1.4878 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   >= 0
 intermediate_flow_acetic_anhydride_upper_bound:
   flow[acetic_anhydride,_from_acetaldehyde_by_direct_oxidation]
   - 1.4878 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   <= 0.01
 intermediate_flow_syngas: flow[syngas,_from_syngas_bs]
   + flow[syngas,_from_syngas_bl]
   - 6.220493617021276 flow[hydrogen,_from_partial_condensation_of_syngas]
   - 0.5187288 flow[carbon_monoxide,_from_partial_condensation_of_syngas]
   >= 0
 intermediate_flow_syngas_upper_bound: flow[syngas,_from_syngas_bs]
   + flow[syngas,_from_syngas_bl]
   - 6.220493617021276 flow[hydrogen,_from_partial_condensation_of_syngas]
   - 0.5187288 flow[carbon_monoxide,_from_partial_condensation_of_syngas]
   <= 0.01
 intermediate_flow_syngas_bs:
   flow[syngas,_from_agricultural_residue_gasification]
   - flow[syngas,_from_syngas_bs] >= 0
 intermediate_flow_syngas_bs_upper_bound:
   flow[syngas,_from_agricultural_residue_gasification]
   - flow[syngas,_from_syngas_bs] <= 0.01
 intermediate_flow_syngas_bl:
   flow[syngas,_from_forest_residue_gasification]
   - flow[syngas,_from_syngas_bl] >= 0
 intermediate_flow_syngas_bl_upper_bound:
   flow[syngas,_from_forest_residue_gasification]
   - flow[syngas,_from_syngas_bl] <= 0.01
 intermediate_flow_carbon_monoxide:
   flow[carbon_monoxide,_from_partial_condensation_of_syngas]
   - 0.4418269 flow[acetic_acid,_from_methanol_carbonylation] >= 0
 intermediate_flow_carbon_monoxide_upper_bound:
   flow[carbon_monoxide,_from_partial_condensation_of_syngas]
   - 0.4418269 flow[acetic_acid,_from_methanol_carbonylation] <= 0.01
 intermediate_flow_acetic_acid:
   flow[acetic_acid,_from_methanol_carbonylation]
   - 0.772 flow[sodium_acetate,_from_sodium_hydroxide_and_acetic_acid]
   + 1.19214 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   - 0.05552 flow[terephthalic_acid,_from_p-xylene_oxidation] >= 0
 intermediate_flow_sodium_acetate:
   flow[sodium_acetate,_from_sodium_hydroxide_and_acetic_acid]
   - 0.0408 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   >= 0
 intermediate_flow_sodium_acetate_upper_bound:
   flow[sodium_acetate,_from_sodium_hydroxide_and_acetic_acid]
   - 0.0408 flow[cellulose_acetate,_from_celllulose_and_acetic_anhydride]
   <= 0.01
 intermediate_flow_vinyl_chloride:
   flow[vinyl_chloride,_from_ethylene_chlorination]
   - 1.008 flow[PVC,_from_bulk_polymerization] >= 0
 intermediate_flow_vinyl_chloride_upper_bound:
   flow[vinyl_chloride,_from_ethylene_chlorination]
   - 1.008 flow[PVC,_from_bulk_polymerization] <= 0.01
 intermediate_flow_ethylene_glycol:
   flow[ethylene_glycol,_from_ethylene_and_oxygen]
   - 0.31887 flow[PET,_from_ethylene_glycol_and_terephthalic_acid] >= 0
 intermediate_flow_ethylene_glycol_upper_bound:
   flow[ethylene_glycol,_from_ethylene_and_oxygen]
   - 0.31887 flow[PET,_from_ethylene_glycol_and_terephthalic_acid] <= 0.01
 intermediate_flow_terephthalic_acid:
   flow[terephthalic_acid,_from_p-xylene_oxidation]
   - 0.85399 flow[PET,_from_ethylene_glycol_and_terephthalic_acid] >= 0
 intermediate_flow_terephthalic_acid_upper_bound:
   flow[terephthalic_acid,_from_p-xylene_oxidation]
   - 0.85399 flow[PET,_from_ethylene_glycol_and_terephthalic_acid] <= 0.01
 intermediate_flow_ethylbenzene:
   flow[ethylbenzene,_from_liquid_phase_alkylation]
   + flow[ethylbenzene,_from_butadiene]
   - 1.06 flow[styrene,_from_ethylbenzene]
   + 0.1905 flow[styrene,_from_toluene_and_methanol] >= 0
 intermediate_flow_ethylbenzene_upper_bound:
   flow[ethylbenzene,_from_liquid_phase_alkylation]
   + flow[ethylbenzene,_from_butadiene]
   - 1.06 flow[styrene,_from_ethylbenzene]
   + 0.1905 flow[styrene,_from_toluene_and_methanol] <= 0.01
 intermediate_flow_hydrogen_peroxide:
   flow[hydrogen_peroxide,_from_anthraquinone_process]
   - 0.0840197693574959 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - 0.0840197693574959 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - 0.0840197693574959 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - 0.0840197693574959 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - 0.0840197693574959 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - 0.0840197693574959 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   >= 0
 intermediate_flow_hydrogen_peroxide_upper_bound:
   flow[hydrogen_peroxide,_from_anthraquinone_process]
   - 0.0840197693574959 flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - 0.0840197693574959 flow[ethylene,_from_ethanol_bs_to_butadiene_process]
   - 0.0840197693574959 flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - 0.0840197693574959 flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - 0.0840197693574959 flow[ethylene,_from_ethanol_bl_to_butadiene_process]
   - 0.0840197693574959 flow[propylene,_from_ethanol_bl_to_butadiene_process]
   <= 0.01
 intermediate_flow_butadiene_bs:
   flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - flow[butadiene,_from_butadiene_bs] >= 0
 intermediate_flow_butadiene_bs_upper_bound:
   flow[butadiene,_from_ethanol_bs_to_butadiene_process]
   - flow[butadiene,_from_butadiene_bs] <= 0.01
 intermediate_flow_propylene_bs:
   flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - flow[propylene,_from_propylene_bs] >= 0
 intermediate_flow_propylene_bs_upper_bound:
   flow[propylene,_from_ethanol_bs_to_butadiene_process]
   - flow[propylene,_from_propylene_bs] <= 0.01
 intermediate_flow_butadiene_bl:
   flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - flow[butadiene,_from_butadiene_bl] >= 0
 intermediate_flow_butadiene_bl_upper_bound:
   flow[butadiene,_from_ethanol_bl_to_butadiene_process]
   - flow[butadiene,_from_butadiene_bl] <= 0.01
 intermediate_flow_propylene_bl:
   flow[propylene,_from_ethanol_bl_to_butadiene_process]
   - flow[propylene,_from_propylene_bl] >= 0
 intermediate_flow_propylene_bl_upper_bound:
   flow[propylene,_from_ethanol_bl_to_butadiene_process]
   - flow[propylene,_from_propylene_bl] <= 0.01
 intermediate_flow_butadiene: flow[butadiene,_from_butadiene_bs]
   + flow[butadiene,_from_butadiene_bl]
   - 1.04264 flow[ethylbenzene,_from_butadiene]
   - 1.023 flow[polybutadiene,_from_solution_polymerization] >= 0
 intermediate_flow_butadiene_upper_bound:
   flow[butadiene,_from_butadiene_bs] + flow[butadiene,_from_butadiene_bl]
   - 1.04264 flow[ethylbenzene,_from_butadiene]
   - 1.023 flow[polybutadiene,_from_solution_polymerization] <= 0.01
 intermediate_flow_styrene: flow[styrene,_from_ethylbenzene]
   + flow[styrene,_from_benzene_and_ethylene]
   + flow[styrene,_from_toluene_and_methanol]
   - 1.005 flow[GPPS,_from_continuous_bulk_polymerization]
   - 0.9343 flow[HIPS,_from_continuous_bulk_polymerization] >= 0
 intermediate_flow_styrene_upper_bound: flow[styrene,_from_ethylbenzene]
   + flow[styrene,_from_benzene_and_ethylene]
   + flow[styrene,_from_toluene_and_methanol]
   - 1.005 flow[GPPS,_from_continuous_bulk_polymerization]
   - 0.9343 flow[HIPS,_from_continuous_bulk_polymerization] <= 0.01
 intermediate_flow_polybutadiene:
   flow[polybutadiene,_from_solution_polymerization]
   - 0.06 flow[HIPS,_from_continuous_bulk_polymerization] >= 0
 intermediate_flow_polybutadiene_upper_bound:
   flow[polybutadiene,_from_solution_polymerization]
   - 0.06 flow[HIPS,_from_continuous_bulk_polymerization] <= 0.01
Bounds
End
