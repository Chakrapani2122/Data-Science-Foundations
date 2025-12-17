import pandas as pd
import numpy as np
from scipy.stats import t, ttest_ind_from_stats

# =====================================================
# 1. FILE PATH
# =====================================================
file_path = "statsresults.xlsx"

# =====================================================
# 2. LOAD EXCEL SAFELY
# =====================================================
df = pd.read_excel(file_path, engine="openpyxl")

# =====================================================
# 3. CLEAN NUMERIC COLUMNS (EXCEL IS DIRTY)
# =====================================================
numeric_cols = [
    "Class 1 Mean",
    "Class 1 Standard Error",
    "Class 2 mean",
    "Class 2 Standard Error"
]

for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =====================================================
# 4. SAMPLE SIZES
# =====================================================
n1 = 1150   # Low artists
n2 = 1142   # Top artists

# =====================================================
# 5. WELCH T-TEST (MANUAL, FORMULA BASED)
# =====================================================
def welch_ttest_manual(row):
    m1 = row["Class 1 Mean"]
    m2 = row["Class 2 mean"]
    se1 = row["Class 1 Standard Error"]
    se2 = row["Class 2 Standard Error"]

    if pd.isna(m1) or pd.isna(m2) or pd.isna(se1) or pd.isna(se2):
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

    # t statistic
    t_stat = (m1 - m2) / np.sqrt(se1**2 + se2**2)

    # Welch degrees of freedom
    df_welch = (se1**2 + se2**2)**2 / (
        (se1**4) / (n1 - 1) + (se2**4) / (n2 - 1)
    )

    # two-tailed p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df_welch))

    return pd.Series([t_stat, df_welch, p_value, "Welch"])

# =====================================================
# 6. SCIPY CROSS-CHECK (UDAT EQUIVALENT)
# =====================================================
def welch_ttest_scipy(row):
    m1 = row["Class 1 Mean"]
    m2 = row["Class 2 mean"]
    se1 = row["Class 1 Standard Error"]
    se2 = row["Class 2 Standard Error"]

    if pd.isna(m1) or pd.isna(m2) or pd.isna(se1) or pd.isna(se2):
        return pd.Series([np.nan, np.nan])

    # Convert SE → SD
    sd1 = se1 * np.sqrt(n1)
    sd2 = se2 * np.sqrt(n2)

    t_stat, p_val = ttest_ind_from_stats(
        mean1=m1,
        std1=sd1,
        nobs1=n1,
        mean2=m2,
        std2=sd2,
        nobs2=n2,
        equal_var=False
    )

    return pd.Series([t_stat, p_val])

# =====================================================
# 7. APPLY TESTS
# =====================================================
df[["t_manual", "df_welch", "p_manual", "test_type"]] = df.apply(
    welch_ttest_manual, axis=1
)

df[["t_scipy", "p_scipy"]] = df.apply(
    welch_ttest_scipy, axis=1
)

# =====================================================
# 8. SANITY CHECK DIFFERENCE
# =====================================================
df["t_diff"] = df["t_manual"] - df["t_scipy"]
df["p_diff"] = df["p_manual"] - df["p_scipy"]

# =====================================================
# 9. SAVE BACK INTO SAME EXCEL FILE
# =====================================================
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name="Welch_T_Test_Results", index=False)

# =====================================================
# 10. DONE
# =====================================================
print("✔ Welch t-test completed and saved into statsresults.xlsx")
