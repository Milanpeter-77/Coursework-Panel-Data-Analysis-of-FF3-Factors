# Import libraries -----
import pandas as pd
import numpy as np
import nbformat


import scipy.stats as st
import statsmodels.api as sm

import linearmodels.panel as lmp



# Read data -----
dfPRICES = pd.read_csv('capmff/capmff_2010-2025_prices.csv')
dfSECTOR = pd.read_csv('capmff/capmff_2010-2025_sector.csv')
dfFACTORS = pd.read_csv('capmff/capmff_2010-2025_ff.csv')

# Data preparation -----

# Ensure dates are datetime format
dfFACTORS['Date'] = pd.to_datetime(dfFACTORS['Date'])
dfPRICES['Date'] = pd.to_datetime(dfPRICES['Date'])

# Ensure price dates match factor dates
# Note: price dates are aligned to factor dates,
# because the last trading day is later than the last factor date
dfPRICES = dfPRICES[dfPRICES['Date'].isin(dfFACTORS['Date'])]

# Drop unnecessary columns
dfSECTOR = dfSECTOR.drop(columns=['industry', 'country', 'employees'])

# Rename factor columns for consistency
dfFACTORS = dfFACTORS.rename(columns={'Mkt-RF': 'MKT'})

# Rename sector values for consistency
dfSECTOR['sector'] = dfSECTOR['sector'].replace({'Energy': 'ENERGY',
                                                 'Financial Services': 'FINSERVICES',
                                                 'Healthcare': 'HEALTHCARE'})

# List required sectors, filter and get tickers
lSECTORS = ['ENERGY', 'FINSERVICES', 'HEALTHCARE']
dfSECTOR = dfSECTOR[dfSECTOR['sector'].isin(lSECTORS)]
lTICKERS = dfSECTOR['Ticker'].tolist()

# Filter prices data
dfPRICES = dfPRICES[['Date'] + lTICKERS]

# List factors
lFACTORS = ['MKT', 'SMB', 'HML']

# Separate risk free rate and factors data
dfRF = dfFACTORS[['Date', 'RF']]
dfFACTORS = dfFACTORS[['Date'] + lFACTORS]

# Create excess returns dataframe
dfRETURNS = dfPRICES.copy()
dfRETURNS[lTICKERS] = dfPRICES[lTICKERS].pct_change(fill_method=None).sub(dfRF['RF'], axis=0)

# Panel data structure -----

# Create panel data in long format
dfPANEL = dfRETURNS.melt(id_vars=['Date'], value_vars=lTICKERS, var_name='Ticker', value_name='EXCESSRETURN')
# Drop missing excess returns values
dfPANEL = dfPANEL.dropna(subset=['EXCESSRETURN']).reset_index(drop=True)

# Merge with factors data
dfPANEL = dfPANEL.merge(dfFACTORS, on='Date', how='left')
# Drop rows with missing factor values
dfPANEL = dfPANEL.dropna(subset=lFACTORS).reset_index(drop=True)

# Add sector dummy variables
dfPANEL = dfPANEL.merge(dfSECTOR, on='Ticker', how='left')
dfPANEL = pd.get_dummies(dfPANEL, columns=['sector'], drop_first=False, prefix='', prefix_sep='')


# Set multi-index (linearmodels expects (entity, time) order)
dfPANEL = dfPANEL.set_index(['Ticker', 'Date']).sort_index()


# Create sector-factor interaction terms
lINTERACTIONS = []
for s in lSECTORS:
    for f in lFACTORS:
        dfPANEL[f'{f}_{s}'] = dfPANEL[f] * dfPANEL[s]
        lINTERACTIONS.append(f'{f}_{s}')

# Equation 3.1. Estimation -----

# Pooled OLS Regression
mdlPOOLED = sm.OLS(endog=dfPANEL['EXCESSRETURN'],
                   exog=sm.add_constant(dfPANEL[lFACTORS])).fit()

# Display results
print(mdlPOOLED.summary())

# Export to LaTeX
with open('documentation/output/pooled-ols.tex', 'w') as f:
    f.write(mdlPOOLED.summary().as_latex())

# Equation 3.2. Estimation -----

# Sector-specific Pooled OLS Regression
mdlINTER = sm.OLS(endog=dfPANEL['EXCESSRETURN'],
                  exog=(dfPANEL[lINTERACTIONS])).fit()

# Display results
print(mdlINTER.summary())

# Export to LaTeX
with open('documentation/output/sector-specific-ols.tex', 'w') as f:
    f.write(mdlINTER.summary().as_latex())

# Equation 3.3. Estimation -----

# Firm Fixed Effects model using Within Transformation with clustered standard errors
mdlFE = lmp.PanelOLS(dependent=dfPANEL['EXCESSRETURN'],
                         exog=sm.add_constant(dfPANEL[lFACTORS]),
                         entity_effects=True).fit(cov_type="clustered", cluster_entity=True)

# Display results
print (mdlFE.summary)

# Export to LaTeX
with open('documentation/output/fixed-effects.tex', 'w') as f:
    f.write(mdlFE.summary.as_latex())

# Equation 3.4. Estimation -----

# Firm Random Effects model with clustered standard errors
mdlRE = lmp.RandomEffects(dependent=dfPANEL['EXCESSRETURN'],
                          exog=sm.add_constant(dfPANEL[lFACTORS])).fit(cov_type="clustered", cluster_entity=True)

# Display results
print (mdlRE.summary)

# Export to LaTeX
with open('documentation/output/random-effects.tex', 'w') as f:
    f.write(mdlRE.summary.as_latex())

# Model statistic comparison table -----

# Collect statistics for each model
dtMODELSTATS = {
    "Pooled OLS": {
        "R-squared": mdlPOOLED.rsquared,
        "Adj. R-squared": mdlPOOLED.rsquared_adj,
        "R-squared (Within)": np.nan,
        "R-squared (Between)": np.nan,
        "R-squared (Overall)": np.nan,
        "Log-Likelihood": mdlPOOLED.llf,
        "AIC": mdlPOOLED.aic,
        "BIC": mdlPOOLED.bic,
        "F-statistic": mdlPOOLED.fvalue,
        "Prob(F-statistic)": mdlPOOLED.f_pvalue,
        "Observations": int(mdlPOOLED.nobs),
        "Entities": np.nan,
        "Time Periods": np.nan,
        "Estimator/Notes": "OLS"
    },
    "Interaction OLS": {
        "R-squared": mdlINTER.rsquared,
        "Adj. R-squared": mdlINTER.rsquared_adj,
        "R-squared (Within)": np.nan,
        "R-squared (Between)": np.nan,
        "R-squared (Overall)": np.nan,
        "Log-Likelihood": mdlINTER.llf,
        "AIC": mdlINTER.aic,
        "BIC": mdlINTER.bic,
        "F-statistic": mdlINTER.fvalue,
        "Prob(F-statistic)": mdlINTER.f_pvalue,
        "Observations": int(mdlINTER.nobs),
        "Entities": np.nan,
        "Time Periods": np.nan,
        "Estimator/Notes": "OLS with interactions"
    },
    "Fixed Effects": {
        "R-squared": mdlFE.rsquared,
        "Adj. R-squared": np.nan,
        "R-squared (Within)": mdlFE.rsquared_within,
        "R-squared (Between)": mdlFE.rsquared_between,
        "R-squared (Overall)": mdlFE.rsquared_overall,
        "Log-Likelihood": mdlFE.loglik,
        "AIC": getattr(mdlFE, "aic", np.nan),
        "BIC": getattr(mdlFE, "bic", np.nan),
        "F-statistic": mdlFE.f_statistic.stat if mdlFE.f_statistic is not None else np.nan,
        "Prob(F-statistic)": mdlFE.f_statistic.pval if mdlFE.f_statistic is not None else np.nan,
        "Observations": int(mdlFE.nobs),
        "Entities": mdlFE.entity_info['total'],
        "Time Periods": mdlFE.time_info['total'],
        "Estimator/Notes": "PanelOLS, clustered SEs"
    },
    "Random Effects": {
        "R-squared": mdlRE.rsquared,
        "Adj. R-squared": np.nan,
        "R-squared (Within)": mdlRE.rsquared_within,
        "R-squared (Between)": mdlRE.rsquared_between,
        "R-squared (Overall)": mdlRE.rsquared_overall,
        "Log-Likelihood": mdlRE.loglik,
        "AIC": getattr(mdlRE, "aic", np.nan),
        "BIC": getattr(mdlRE, "bic", np.nan),
        "F-statistic": mdlRE.f_statistic.stat if mdlRE.f_statistic is not None else np.nan,
        "Prob(F-statistic)": mdlRE.f_statistic.pval if mdlRE.f_statistic is not None else np.nan,
        "Observations": int(mdlRE.nobs),
        "Entities": mdlRE.entity_info['total'],
        "Time Periods": mdlRE.time_info['total'],
        "Estimator/Notes": "RandomEffects, clustered SEs"
    }
}

# Round numeric values for better display
for m in dtMODELSTATS:
    for k in dtMODELSTATS[m]:
        if isinstance(dtMODELSTATS[m][k], float):
            dtMODELSTATS[m][k] = round(dtMODELSTATS[m][k], 4)

# Convert to DataFrame
dfMODELSTATS = pd.DataFrame(dtMODELSTATS)

# Display
display(dfMODELSTATS)

# Export to LaTeX
strMODELSTATS = dfMODELSTATS.to_latex(
    # 3 significant digits, scientific if large
    float_format=lambda x: f"{x:.3g}",  
    na_rep="--",
    column_format="lcccc",
    caption=None)

with open("documentation/output/model_comparison_stats.tex", "w") as f:
    f.write(strMODELSTATS)

# Model parameters comparison table -----

lROW_ORDER = [
    "const", "MKT", "SMB", "HML",
    "MKT_ENERGY", "SMB_ENERGY", "HML_ENERGY",
    "MKT_FINSERVICES", "SMB_FINSERVICES", "HML_FINSERVICES",
    "MKT_HEALTHCARE", "SMB_HEALTHCARE", "HML_HEALTHCARE"]


dtMODELPARAMS = {}

for p in lROW_ORDER:
    dtMODELPARAMS[p] = {
        "Pooled OLS": (
            f"{mdlPOOLED.params.get(p, np.nan):.4f} "
            f"({mdlPOOLED.bse.get(p, np.nan):.4f})"
            if p in mdlPOOLED.params else np.nan
        ),
        "Interaction OLS": (
            f"{mdlINTER.params.get(p, np.nan):.4f} "
            f"({mdlINTER.bse.get(p, np.nan):.4f})"
            if p in mdlINTER.params else np.nan
        ),
        "Fixed Effects": (
            f"{mdlFE.params.get(p, np.nan):.4f} "
            f"({mdlFE.std_errors.get(p, np.nan):.4f})"
            if p in mdlFE.params else np.nan
        ),
        "Random Effects": (
            f"{mdlRE.params.get(p, np.nan):.4f} "
            f"({mdlRE.std_errors.get(p, np.nan):.4f})"
            if p in mdlRE.params else np.nan
        ),
    }

# Convert to DataFrame (and transpose for correct orientation)
dfMODELPARAMS = pd.DataFrame(dtMODELPARAMS).T

# Display
display(dfMODELPARAMS)

# Export to LaTeX
strMODELPARAMS = dfMODELPARAMS.to_latex(
    na_rep="--",
    column_format="lcccc",
    caption=None,
    escape=True)

with open("documentation/output/model_comparison_params.tex", "w") as f:
    f.write(strMODELPARAMS)

# F-test: Pooled OLS vs Pooled OLS with Sector-Specific Slopes (Interaction) -----

# Residual sums of squares
dRRSS = float(np.sum(mdlPOOLED.resid**2)) # restricted (pooled)
dURSS = float(np.sum(mdlINTER.resid**2)) # unrestricted (interaction)

iM  = dfPANEL.index.get_level_values('Ticker').nunique()
iNT = dfPANEL.shape[0]
iK  = len(lFACTORS)

iDoF1 = iM - 1
iDoF2 = iNT - iM - iK

dF_stat = ((dRRSS - dURSS) / iDoF1) / (dURSS / iDoF2)
dP_val  = st.f.sf(dF_stat, iDoF1, iDoF2)

print(f"F-test (Pooled OLS vs Sector-Specific Slopes): F = {dF_stat:.4f}, df1={iDoF1}, df2={iDoF2}, p-value={dP_val:.3g}")

if dP_val < 0.05:
    print("Reject H0: Sector-specific slopes significantly improve the model.")
else:
    print("Do not reject H0: No significant gain from sector-specific slopes.")

# F-test: Pooled OLS vs Fixed Effects -----

# Residual sums of squares
dRRSS = float(np.sum(mdlPOOLED.resid**2)) # restricted (pooled)
dURSS = float(np.sum(mdlFE.resids**2)) # unrestricted (fixed effects)

iM  = dfPANEL.index.get_level_values('Ticker').nunique()
iNT = dfPANEL.shape[0]
iK  = len(lFACTORS)

iDoF1 = iM - 1
iDoF2 = iNT - iM - iK

dF_stat = ((dRRSS - dURSS) / iDoF1) / (dURSS / iDoF2)
dP_val  = st.f.sf(dF_stat, iDoF1, iDoF2)

print(f"F-test (Pooled OLS vs Fixed Effects): F = {dF_stat:.4f}, df1={iDoF1}, df2={iDoF2}, p-value={dP_val:.3g}")

if dP_val < 0.05:
    print("Reject H0: Firm fixed effects matter. Prefer FE over pooled OLS.")
else:
    print("Do not reject H0: No evidence that fixed effects are needed.")

# Hausman test (Fixed Effects vs Random Effects) -----

# Extract comparable slope vectors & covariance blocks
db_fe = mdlFE.params.loc[lFACTORS]
dV_fe = mdlFE.cov.loc[lFACTORS, lFACTORS]

db_re = mdlRE.params.loc[lFACTORS]
dV_re = mdlRE.cov.loc[lFACTORS, lFACTORS]

db_diff = (db_fe - db_re).values
dV_diff = (dV_fe - dV_re).values

# Numerical safeguard: use pseudo-inverse in case V_diff is near-singular
dH = float(db_diff.T @ np.linalg.pinv(dV_diff) @ db_diff)
dDoF = len(lFACTORS)
dP_val = st.chi2.sf(dH, dDoF)

print(f"Hausman test (FE vs RE): H = {dH:.4f}, df = {dDoF}, p-value = {dP_val:.3g}")
if dP_val < 0.05:
    print("Reject H0: Random Effects model inconsistent. Prefer Fixed Effects.")
else:
    print("Do not reject H0: Random Effects is consistent and efficient.")

# Export python code / markdown cells separately


# Load notebook
nb = nbformat.read("eqf-assignment-1-script.ipynb", as_version=4)

# Write code cells into one python script
with open("documentation/eqf-1-script.py", "w", encoding="utf-8") as f:
    for cell in nb.cells:
        if cell.cell_type == "code":
            f.write(cell.source.replace("-", "-") + "\n\n")

# Collect markdown cells
md_cells = [cell['source'] for cell in nb.cells if cell['cell_type'] == 'markdown']

# Write them into one markdown file
with open("documentation/eqf-1-markdown.md", "w") as f:
    f.write("\n\n".join(md_cells))

# Convert markdown to LaTeX using pandoc
# run in bash
# pandoc documentation/eqf-1-markdown.md -o documentation/eqf-1-markdown.tex



