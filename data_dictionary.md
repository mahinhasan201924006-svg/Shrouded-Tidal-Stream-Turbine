# Data dictionary

Both CSV files share the same schema. `Turbine_LHS_Data_64_ML_completed.csv` is the cleaned
file used for surrogate training.

| Column | Description | Units |
|--------|-------------|-------|
| `Name` | Design-point identifier (`DP 0` … `DP 63`) | — |
| `P1_chord_ratio_c_by_D` | Duct chord-to-rotor-diameter ratio, c/D | — |
| `P2_AoA_angle_deg` | Duct angle of attack, α | degrees |
| `P3_Y_offset_m` | Radial offset (duct inner wall to rotor-tip plane), y | m |
| `P4_Pressure_Jump_Pa` | Actuator-disk pressure jump, ΔP | Pa |
| `P5_drag_duct_N` | Duct axial force, F_d (Fluent sign convention; magnitude used as load) | N |
| `P6_u_disk_m_per_s` | Disk-averaged axial velocity, u_d | m s⁻¹ |
| `P7` | Exported CFD rotor-area-normalized power coefficient, C_P,r | — |
| `Data_Status` | `Successful` (converged, n=61) or `Failed` (non-converged, n=3) | — |

Fixed parameters: rotor diameter D = 1.0 m (A_r = π/4 = 0.785 m²); freestream U∞ = 1.0 m s⁻¹;
water density ρ = 998.2 kg m⁻³. The physics-reconstructed coefficient is
C_P,r = ΔP·u_d / (0.5·ρ·U∞³) = ΔP·u_d / 499.1.

Failed/non-converged rows are retained for traceability and are excluded from surrogate
training and from all CFD-validated performance claims.
