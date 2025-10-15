# **Panel Data Analysis of Fama–French Factor Models Across Sectors**

## **General Description of the Work**

This assignment examined how firm returns within the S&P 500 relate to the Fama–French three factors using panel data econometrics, comparing different model specifications and testing whether factor sensitivities differ across sectors.

## **Data Preparation and Setup**

I used daily stock price data, sector classifications, and Fama–French factor data for S&P 500 firms. After aligning dates and converting prices to daily excess returns, I filtered the dataset to the assigned sectors — **Energy, Financial Services, and Healthcare** — and merged in sector identifiers and factor variables to create a clean, long-format panel suitable for regression analysis.

## **Model Estimation**

I estimated multiple models to capture different forms of firm and sector heterogeneity:

* **Pooled OLS Regression:** estimated a single set of coefficients for all firms as a benchmark.
* **Pooled OLS with Sector Interactions:** allowed factor loadings to differ by sector via interaction terms between factors and sector dummies.
* **Fixed Effects (FE):** controlled for unobserved firm-specific heterogeneity using the within transformation.
* **Random Effects (RE):** modelled firm-specific effects as random variables under the assumption of no correlation with regressors.

Each model was estimated with clustered standard errors at the firm level to ensure valid inference under potential serial correlation and heteroskedasticity.

## **Model Comparison and Statistical Tests**

I compared the models based on **R²**, **AIC/BIC**, **log-likelihood**, and **robust F-statistics**. I also conducted a series of tests:

* **F-tests** to compare nested models (pooled vs. sector-interacted, pooled vs. fixed effects).
* **Hausman test** to decide between fixed and random effects.

The F-test supported the inclusion of firm-specific effects, while the Hausman test indicated that random effects were consistent and efficient.

## **Conclusions**

All models yielded highly significant factor loadings with stable coefficients across specifications. Sector interactions revealed meaningful but not jointly significant differences in factor sensitivities. The **Random Effects model** was ultimately preferred for its efficiency and ability to exploit both within- and between-firm variation, while maintaining robust inference through clustered standard errors.
