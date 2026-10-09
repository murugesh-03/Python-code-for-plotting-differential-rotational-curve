"""
Solar Differential Rotation Curve from Sunspot Tracking Data
--------------------------------------------------------------

Reads:
    NOAA_xxxxx_Sunspot_Rotation.xlsx

For each sunspot group:
    - Mean heliographic latitude
    - Synodic rotation rate

The synodic rotation rate is converted to
sidereal rotation rate.

The differential rotation law is fitted using
the LEAST-SQUARES METHOD:

    omega(phi) = A + B*sin^2(phi) + C*sin^4(phi)

IMPORTANT:
    No standard rotation model is used.
    No Snodgrass & Ulrich curve is used.
    No scipy is required.
"""


# ==============================================================
# IMPORT LIBRARIES
# ==============================================================

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog


# ==============================================================
# 1. SELECT FOLDER
# ==============================================================

root = Tk()
root.withdraw()

DATA_FOLDER = filedialog.askdirectory(
    title="Select folder containing NOAA Sunspot Excel files"
)

root.destroy()


if not DATA_FOLDER:

    print("No folder selected.")

    input("Press Enter to exit...")

    raise SystemExit


FILE_PATTERN = "NOAA_*_Sunspot_Rotation.xlsx"


# Earth's orbital angular velocity
EARTH_ORBIT_RATE = 360.0 / 365.25


print("\nSelected folder:")
print(DATA_FOLDER)


# ==============================================================
# 2. READ EACH EXCEL FILE
# ==============================================================

def read_sunspot_file(path):

    """
    Returns:

        NOAA number
        Mean latitude
        Synodic rotation rate
    """

    # Read Excel workbook
    df = pd.read_excel(
        path,
        sheet_name=0,
        header=0
    )


    # ----------------------------------------------------------
    # NOAA NUMBER
    # ----------------------------------------------------------

    if "NOAA" in df.columns:

        noaa_values = df["NOAA"].dropna()

        if len(noaa_values) > 0:

            noaa_id = str(
                noaa_values.iloc[0]
            )

        else:

            noaa_id = os.path.basename(path)

    else:

        noaa_id = os.path.basename(path)


    # ----------------------------------------------------------
    # LATITUDE
    # ----------------------------------------------------------

    if "LATITUDE (deg)" not in df.columns:

        print(
            "\nLATITUDE (deg) column not found:"
        )

        print(path)

        return None


    latitude_values = pd.to_numeric(
        df["LATITUDE (deg)"],
        errors="coerce"
    )


    latitude_values = latitude_values.dropna()


    if len(latitude_values) == 0:

        print(
            "\nNo latitude values found:"
        )

        print(path)

        return None


    # Mean heliographic latitude

    mean_lat = latitude_values.mean()


    # ----------------------------------------------------------
    # FIND SYNODIC ROTATION RATE
    # ----------------------------------------------------------

    synodic_rate = None


    for row_number in range(len(df)):

        row_values = df.iloc[
            row_number
        ].values


        row_text = " ".join(

            str(value)

            for value in row_values

            if not pd.isna(value)

        )


        if "Synodic rotation rate" in row_text:

            for value in row_values:

                try:

                    number = float(value)

                    if 0 < number < 20:

                        synodic_rate = number

                        break

                except (ValueError, TypeError):

                    continue


        if synodic_rate is not None:

            break


    # ----------------------------------------------------------
    # CHECK SYNODIC RATE
    # ----------------------------------------------------------

    if synodic_rate is None:

        print(
            "\nSynodic rotation rate not found:"
        )

        print(path)

        return None


    return (
        noaa_id,
        mean_lat,
        synodic_rate
    )


# ==============================================================
# 3. FIND ALL EXCEL FILES
# ==============================================================

file_list = sorted(

    glob.glob(

        os.path.join(
            DATA_FOLDER,
            FILE_PATTERN
        )

    )

)


print(
    "\nNumber of Excel files found:",
    len(file_list)
)


if len(file_list) == 0:

    print(
        "\nNo matching Excel files found."
    )

    print(
        "\nExpected file format:"
    )

    print(
        "NOAA_xxxxx_Sunspot_Rotation.xlsx"
    )

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


# ==============================================================
# 4. READ ALL SUNSPOT DATA
# ==============================================================

records = []


for path in file_list:

    result = read_sunspot_file(path)


    if result is None:

        continue


    noaa_id, mean_lat, synodic_rate = result


    # ----------------------------------------------------------
    # SYNODIC TO SIDEREAL
    # ----------------------------------------------------------

    sidereal_rate = (

        synodic_rate
        + EARTH_ORBIT_RATE

    )


    records.append({

        "NOAA": noaa_id,

        "latitude_deg": mean_lat,

        "synodic_rate_deg_per_day":
            synodic_rate,

        "sidereal_rate_deg_per_day":
            sidereal_rate

    })


# ==============================================================
# 5. CHECK DATA
# ==============================================================

if len(records) < 3:

    print(
        "\nAt least 3 sunspot groups are required."
    )

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


# ==============================================================
# 6. CREATE DATAFRAME
# ==============================================================

data = pd.DataFrame(records)


# Sort according to latitude

data = data.sort_values(
    "latitude_deg"
).reset_index(drop=True)


print("\n")
print("=" * 70)
print("SUNSPOT ROTATION DATA")
print("=" * 70)


print(
    data.to_string(index=False)
)


# ==============================================================
# 7. DIFFERENTIAL ROTATION EQUATION
# ==============================================================

def rotation_law(
    phi_deg,
    A,
    B,
    C
):

    phi = np.radians(phi_deg)

    return (

        A
        + B * np.sin(phi) ** 2
        + C * np.sin(phi) ** 4

    )


# ==============================================================
# 8. OBSERVED DATA
# ==============================================================

lat = data[
    "latitude_deg"
].values


rate = data[
    "sidereal_rate_deg_per_day"
].values


# ==============================================================
# 9. LEAST-SQUARES FIT
#
# Equation:
#
# omega(phi) =
# A + B*sin²(phi) + C*sin⁴(phi)
#
# =============================================================

phi = np.radians(lat)


# Design matrix

X = np.column_stack(

    (

        np.ones(len(phi)),

        np.sin(phi) ** 2,

        np.sin(phi) ** 4

    )

)


# --------------------------------------------------------------
# LEAST-SQUARES SOLUTION
# --------------------------------------------------------------

coefficients, residuals, rank, singular_values = np.linalg.lstsq(

    X,

    rate,

    rcond=None

)


# Extract A, B and C

A = coefficients[0]

B = coefficients[1]

C = coefficients[2]


# ==============================================================
# 10. CALCULATE FITTED VALUES
# ==============================================================

fitted_rate = rotation_law(

    lat,

    A,
    B,
    C

)


# ==============================================================
# 11. CALCULATE R-SQUARED
# ==============================================================

SS_res = np.sum(

    (
        rate
        - fitted_rate
    ) ** 2

)


SS_tot = np.sum(

    (
        rate
        - np.mean(rate)
    ) ** 2

)


if SS_tot != 0:

    R_squared = (

        1
        - SS_res / SS_tot

    )

else:

    R_squared = np.nan


# ==============================================================
# 12. PRINT LEAST-SQUARES RESULTS
# ==============================================================

print("\n")
print("=" * 70)
print("LEAST-SQUARES DIFFERENTIAL ROTATION RESULTS")
print("=" * 70)


print("\nDifferential rotation equation:")


print(

    f"ω(φ) = {A:.4f}"
    f" + ({B:.4f}) sin²φ"
    f" + ({C:.4f}) sin⁴φ"

)


print("\nFitted parameters:")


print(
    f"A = {A:.4f} deg/day"
)


print(
    f"B = {B:.4f} deg/day"
)


print(
    f"C = {C:.4f} deg/day"
)


print(
    f"\nR² = {R_squared:.4f}"
)


# ==============================================================
# 13. CREATE SMOOTH LEAST-SQUARES CURVE
# ==============================================================

minimum_latitude = min(

    -40,

    np.min(lat) - 5

)


maximum_latitude = max(

    40,

    np.max(lat) + 5

)


phi_smooth = np.linspace(

    minimum_latitude,

    maximum_latitude,

    500

)


# Calculate fitted curve

fit_smooth = rotation_law(

    phi_smooth,

    A,
    B,
    C

)


# ==============================================================
# 14. PLOT
# ==============================================================

plt.figure(

    figsize=(9, 6)

)


# --------------------------------------------------------------
# OBSERVED SUNSPOT POINTS
# --------------------------------------------------------------

plt.scatter(

    lat,

    rate,

    s=60,

    zorder=3,

    label="Observed sunspot groups"

)


# --------------------------------------------------------------
# NOAA LABELS
# --------------------------------------------------------------

for _, row in data.iterrows():

    plt.annotate(

        row["NOAA"],

        (

            row["latitude_deg"],

            row["sidereal_rate_deg_per_day"]

        ),

        textcoords="offset points",

        xytext=(5, 5),

        fontsize=8

    )


# --------------------------------------------------------------
# LEAST-SQUARES DIFFERENTIAL ROTATION CURVE
# --------------------------------------------------------------

plt.plot(

    phi_smooth,

    fit_smooth,

    linewidth=2,

    label=(

        f"Least-squares fit: "

        f"{A:.2f} "

        f"{B:+.2f}sin²φ "

        f"{C:+.2f}sin⁴φ"

    )

)


# ==============================================================
# 15. AXIS LABELS
# ==============================================================

plt.xlabel(
    "Heliographic Latitude (deg)",
    fontsize=12
)


plt.ylabel(
    "Sidereal Rotation Rate (deg/day)",
    fontsize=12
)


plt.title(
    "Solar Differential Rotation Curve",
    fontsize=14
)


# ==============================================================
# 16. DISPLAY R-SQUARED
# ==============================================================

plt.text(

    0.05,

    0.05,

    f"R² = {R_squared:.4f}",

    transform=plt.gca().transAxes,

    fontsize=11

)


# ==============================================================
# 17. LEGEND AND GRID
# ==============================================================

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ==============================================================
# 18. SAVE GRAPH
# ==============================================================

output_plot = os.path.join(

    DATA_FOLDER,

    "differential_rotation_curve_least_squares.png"

)


plt.savefig(

    output_plot,

    dpi=300

)


print("\n")
print("=" * 70)

print(
    "Graph saved successfully:"
)

print(
    output_plot
)

print("=" * 70)


# ==============================================================
# 19. SAVE RESULTS TO EXCEL
# ==============================================================

output_excel = os.path.join(

    DATA_FOLDER,

    "Differential_Rotation_Least_Squares_Results.xlsx"

)


# Add fitted rotation rate

data_output = data.copy()


data_output[
    "fitted_sidereal_rate_deg_per_day"
] = fitted_rate


# --------------------------------------------------------------
# Fit parameter table
# --------------------------------------------------------------

fit_results = pd.DataFrame({

    "Parameter": [

        "A",
        "B",
        "C",
        "R-squared"

    ],

    "Value": [

        A,
        B,
        C,
        R_squared

    ],

    "Unit": [

        "deg/day",
        "deg/day",
        "deg/day",
        ""

    ]

})


# --------------------------------------------------------------
# Write Excel file
# --------------------------------------------------------------

with pd.ExcelWriter(

    output_excel,

    engine="openpyxl"

) as writer:

    data_output.to_excel(

        writer,

        sheet_name="Sunspot Data",

        index=False

    )


    fit_results.to_excel(

        writer,

        sheet_name="Least Squares Fit",

        index=False

    )


print(
    "\nLeast-squares results saved:"
)

print(
    output_excel
)


# ==============================================================
# 20. DISPLAY GRAPH
# ==============================================================

plt.show()


input(
    "\nPress Enter to close..."
)
