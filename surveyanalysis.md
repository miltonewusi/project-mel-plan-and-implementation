# Part 2: Analysis of PartnerOrg Baseline and Midline Evaluation Data (Ghana)

![Summary of results](results_summary.png)

## Data and design

- **Source:** the `dataset` tab of the PartnerOrg workbook. It has 756 households in 10 villages: 374 at baseline and 382 at midline.
- **One arm only.** Every record has `arm = intervention`, so there is no comparison group. All results below are **before-and-after changes in intervention villages**, not causal impact estimates.
- **Repeated cross-sections.** There is no household ID, so baseline and midline households cannot be linked. Village sample sizes also differ between waves (e.g. `bb` 43 → 64, `ee` 42 → 28).
- **Savings (Q132a)** is missing for 190 households (123 at baseline, 67 at midline). It is assumed to be in GHS, as the data dictionary states.
- **Software:** Python (pandas, scipy, statsmodels, scikit-learn). The scripts are included in this folder.

---

## a) and b) Variables created

| Variable | Rule | Check |
| --- | --- | --- |
| `animal_own` | 1 if Q033 (cows) > 0, or Q034 (goats) > 0, or Q035 (sheep) > 0, or Q036 (pigs) > 0; else 0 | Matches the existing column in the dataset for all 756 rows |
| `reg_att` | 1 if Q121 = 3 ("Often") or 4 ("Always"); else 0 | Matches the existing column for all 756 rows |

Poultry, rabbits, ox ploughs and "other animals" are excluded from `animal_own`, as the question specifies.

## c) Descriptive statistics (intervention arm)

| Indicator | Wave | N | Mean / proportion | Std dev | Median |
| --- | --- | --- | --- | --- | --- |
| Animal ownership | Baseline | 374 | 0.511 | 0.501 | n/a |
| Animal ownership | Midline | 382 | 0.668 | 0.472 | n/a |
| Regular meeting attendance | Baseline | 374 | 0.519 | 0.500 | n/a |
| Regular meeting attendance | Midline | 382 | 0.712 | 0.453 | n/a |
| Household savings (GHS) | Baseline | 251 | 228,090 | 630,939 | 80,000 |
| Household savings (GHS) | Midline | 315 | 367,680 | 661,763 | 150,000 |

Savings is very right-skewed (maximum GHS 7,000,000), so the **median** is a better summary than the mean.

---

## d) Household wealth index

### Method

I built the index with **Principal Component Analysis (PCA)** on all 20 variables, the standard approach for asset-based wealth indices. The steps were:

1. **Recode yes/no assets (Q022–Q032).** 1 = Yes and 2 = No in the raw data, so I recoded them to 1 = owns and 0 = does not own.
2. **Transform livestock counts (Q033–Q040) with log(1 + count).** A few households report very large numbers (e.g. 200 poultry), which would otherwise dominate the first component.
3. **Transform savings with log(1 + savings).** This handles the heavy skew.
4. **Fill missing savings.** 190 households (25%) have no savings value. I replaced the gap with the pooled median (GHS 100,000) so these households stay in the index. I did not use wave-specific medians, which would have removed part of any real change.
5. **Standardise** all 20 variables to mean 0 and SD 1.
6. **Extract the first principal component**, fitted once on the pooled baseline and midline data. Both waves are therefore scored on the **same scale**, which is essential for comparing them. The sign is set so that higher scores mean wealthier households.

**Diagnostics.** The first component explains 15.8% of the variance, which is typical for survey asset indices. All 20 loadings are positive (from 0.03 for "other animals" to 0.39 for television, with electricity, bank account and refrigerator also high), so the index behaves sensibly as a wealth measure.

### Wealth index by wave

| | Baseline | Midline | Difference |
| --- | --- | --- | --- |
| Mean index score | −0.159 | 0.156 | **+0.315** |
| Std dev | 1.64 | 1.89 | |

Welch t-test p = 0.015. A Mann-Whitney test gives p = 0.016. With village fixed effects and village-clustered standard errors, p = 0.032.

### Share of "rich" households

**Definition:** a household is "rich" if its score is in the **top 20% of the pooled distribution** (cut-off = 0.94). Using one fixed cut-off for both waves means the share can genuinely change over time.

| | Baseline | Midline | Change |
| --- | --- | --- | --- |
| % rich households | 16.8% | 23.3% | **+6.5 pp** |

| Test | Result |
| --- | --- |
| Chi-square, 2×2 | p = 0.027 |
| Village fixed-effects linear probability model, clustered by village | +6.7 pp, p = 0.015 |
| Village fixed-effects logit, clustered by village | Odds ratio 1.56, p = 0.027 |

**Is it significant? Yes, at the 5% level, overall.** The result holds when I change the cut-off:

| Alternative definition | Baseline | Midline | Village-adjusted p |
| --- | --- | --- | --- |
| Top 40% | 36.1% | 44.0% | 0.001 |
| Above baseline 80th percentile | 20.1% | 27.7% | 0.009 |

**Village level.** The share of rich households rose in 7 of 10 villages, but **no single village shows a significant increase** (smallest Fisher p-values: `ee` 0.054, `kk` 0.064). Village samples are only 18 to 64 households per wave, so these tests have little power. The overall gain is a pooled result, not one driven by a particular village.

**Robustness.** Of the four index variants I tested, the mean wealth gain (village-adjusted) is only significant at 5% in two:

| Index variant | Village-adjusted p (mean index) | Village-adjusted p (% rich) |
| --- | --- | --- |
| Main specification | 0.032 | 0.015 |
| Complete-case savings only (n = 566) | 0.026 | 0.020 |
| Without savings | 0.069 | 0.035 |
| Raw counts, no logs | 0.104 | 0.022 |

The **% rich** result is stable across all variants. The **mean score** result is borderline. I would describe the wealth evidence as *suggestive but weaker* than the evidence for the three outcomes in part (e).

---

## e) Has the project improved animal ownership, meeting attendance and savings?

| Outcome | Baseline | Midline | Change | Test (unadjusted) | Village-adjusted estimate (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Animal ownership | 51.1% | 66.8% | **+15.7 pp** | Chi² = 19.2, p < 0.001 | +15.9 pp (9.0 to 22.9), p < 0.001 |
| Regular meeting attendance | 51.9% | 71.2% | **+19.3 pp** | Chi² = 29.9, p < 0.001 | +19.5 pp (12.9 to 26.1), p < 0.001 |
| Median savings (GHS) | 80,000 | 150,000 | **+70,000** (+87.5%) | Mann-Whitney p < 0.001 | Log-savings ratio 1.88×, p < 0.001 |

Further checks:

- **Village consistency.** Animal ownership rose in **10 of 10** villages (paired village test p = 0.001). Meeting attendance rose in **9 of 10** (p = 0.001); only `hh` fell, from 85% to 77%.
- **Savings.** The bootstrap 95% CI for the median increase is GHS 50,000 to 130,000. The mean also rose, by GHS 139,590 (Welch p = 0.011). The 10%-trimmed mean roughly doubled, from about 110,000 to 209,000.
- **Village adjustment.** All three results survive village fixed effects with clustered standard errors. This matters because households within a village are not independent, and the plain tests above ignore that.

**Conclusion:** yes. All three outcomes improved significantly between baseline and midline. Because there is no control group, I can say they improved *in CFM villages*, not that CFM *caused* the improvement.

---

## f) Conclusions, insights and recommendations

### Conclusions

1. **Participation improved most.** Regular attendance at village meetings rose by about 19 points, and in nearly every village. This fits the CFM theory of change, where participation is the entry point for everything else.
2. **Livelihood assets and savings moved in the right direction.** Animal ownership rose by about 16 points and median savings nearly doubled.
3. **Household wealth rose modestly.** The share of "rich" households grew from about 17% to 23%. The mean-score gain is sensitive to how the index is built, so I would treat it as suggestive.

### Caveats

- **No counterfactual.** Drought, prices, other NGOs or a national programme could also explain these gains. Without a comparison group, the results cannot be attributed to CFM.
- **Differential missingness in savings.** 33% of baseline households have no savings value, against 18% at midline (p < 0.001). If non-reporters differ systematically, the savings gain may be overstated or understated. The wealth index is robust to dropping these households, which is reassuring.
- **Different households in each wave.** Part of the change may come from sample composition. Village sample sizes changed a lot between waves.
- **Overlap between outcomes.** Livestock and savings are components of the wealth index, so the wealth result is not independent evidence from part (e).
- **Reporting.** Savings figures are self-reported and very skewed. Please confirm the currency unit and year with PartnerOrg, since values of this size suggest they may be in old cedis.
- **Only 10 villages.** Village-level conclusions have low statistical power.

### Recommendations

1. **Add a comparison group** (matched or phased-in villages) at the next round, so that impact can be attributed to CFM.
2. **Track the same households** (panel) from baseline, and set sample-size targets per village.
3. **Reduce savings non-response.** Review the question wording, enumerator training and consent scripts, and record "refused" vs "don't know" separately.
4. **Report medians and log-changes for savings,** and report the inequality (e.g. Gini or top-20% share), because a few very rich households heavily affect the mean.
5. **Add qualitative work** to understand why attendance rose, and why it fell in `hh`. Investigate the weaker villages (`hh`, `nn`, `gg` on wealth) before the endline.
6. **Check sustainability at endline and after graduation,** to see whether these gains hold when PartnerOrg support ends.

---

## Reproducing the results

```bash
pip install pandas scipy statsmodels scikit-learn matplotlib openpyxl
python 01_wealth_index_and_tests.py   # wealth index, rich-household tests
python 02_robustness_checks.py        # alternative index specs, cluster-adjusted tests, savings checks
```

Edit the file path at the top of each script to point to your copy of the dataset.
