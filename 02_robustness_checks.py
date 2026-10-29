F='ghana_me_dataset.xlsx'; import pandas as pd, numpy as np; from scipy import stats; import statsmodels.formula.api as smf; from sklearn.decomposition import PCA; from sklearn.preprocessing import StandardScaler; d=pd.read_excel(F,sheet_name='dataset'); d['mid']=(d.survey=='midline').astype(int); d['animal_own']=(d[['Q033','Q034','Q035','Q036']]>0).any(axis=1).astype(int); d['reg_att']=d.Q121.isin([3,4]).astype(int)
import warnings; warnings.filterwarnings('ignore')
exec("bin_vars=['Q022','Q023','Q024','Q025','Q026','Q027','Q028','Q029','Q030','Q031','Q032']\ncnt_vars=['Q033','Q034','Q035','Q036','Q037','Q038','Q039','Q040']")
grp=pd.factorize(d.village)[0]
def build(log=True,sav='impute',inc_sav=True):
    X=pd.DataFrame(index=d.index)
    for v in bin_vars: X[v]=(d[v]==1).astype(int)
    for v in cnt_vars: X[v]=np.log1p(d[v]) if log else d[v]
    if inc_sav:
        s=d.Q132a
        if sav=='impute': s=s.fillna(s.median())
        X['sav']=np.log1p(s) if log else s
    return X
def run(X,mask=None,label=''):
    Z=X if mask is None else X[mask]
    sc=StandardScaler().fit(Z); p=PCA(1).fit(sc.transform(Z)); s=sc.transform(X)@p.components_[0]
    if p.components_[0][list(X.columns).index('Q032')]<0: s=-s
    dd=d.assign(w=s); dd['r']=(dd.w>=dd.w[mask if mask is not None else slice(None)].quantile(.8)).astype(int) if mask is not None else (dd.w>=dd.w.quantile(.8)).astype(int)
    m=smf.ols('w~mid+C(village)',dd).fit(cov_type='cluster',cov_kwds={'groups':grp})
    lp=smf.ols('r~mid+C(village)',dd).fit(cov_type='cluster',cov_kwds={'groups':grp})
    p0=dd.r[dd.mid==0].mean(); p1=dd.r[dd.mid==1].mean()
    t=stats.ttest_ind(dd.w[dd.mid==1],dd.w[dd.mid==0],equal_var=False).pvalue
    print(f'{label:40s} varexp={p.explained_variance_ratio_[0]:.3f} meandiff_p(welch)={t:.3f} FEdiff={m.params["mid"]:.3f} p={m.pvalues["mid"]:.3f} | rich {p0:.3f}->{p1:.3f} LPM p={lp.pvalues["mid"]:.3f}')
run(build(),label='MAIN log+impute')
run(build(),mask=d.Q132a.notna(),label='complete-case savings (n=566)')
run(build(inc_sav=False),label='no savings')
run(build(log=False),label='raw counts+savings (no log)')
X=build(); 
# simple unweighted asset count
d['w_simple']=sum((d[v]==1).astype(int) for v in bin_vars)+ (d[['Q033','Q034','Q035','Q036','Q037','Q038','Q039','Q040']]>0).sum(axis=1)
print('asset count mean',d.groupby('survey').w_simple.mean().round(3).to_dict(), stats.ttest_ind(d.w_simple[d.mid==1],d.w_simple[d.mid==0]).pvalue)
# part e with clustering
for y in ['animal_own','reg_att']:
    t=pd.crosstab(d.survey,d[y]); chi=stats.chi2_contingency(t,correction=False)
    lp=smf.ols(f'{y}~mid+C(village)',d).fit(cov_type='cluster',cov_kwds={'groups':grp})
    lg=smf.logit(f'{y}~mid+C(village)',d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':grp})
    ci=lp.conf_int().loc['mid'].values
    print(y,'chi2',round(chi[0],2),'p',chi[1],'FE-LPM',round(lp.params['mid'],3),ci.round(3),'p',round(lp.pvalues['mid'],4),'OR',round(np.exp(lg.params['mid']),2),'p',round(lg.pvalues['mid'],4))
    # per-village diffs
    print(d.groupby(['village','survey'])[y].mean().unstack().round(2).assign(diff=lambda x:x.midline-x.baseline).T.to_string())
d['lsav']=np.log1p(d.Q132a)
s=d.dropna(subset=['Q132a'])
print('mwu',stats.mannwhitneyu(s.Q132a[s.mid==1],s.Q132a[s.mid==0]))
print('log t',stats.ttest_ind(s.lsav[s.mid==1],s.lsav[s.mid==0],equal_var=False))
m=smf.ols('lsav~mid+C(village)',s).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(s.village)[0]}); print('log FE',m.params['mid'],m.pvalues['mid'],np.exp(m.params['mid']))
q=smf.quantreg('Q132a~mid+C(village)',s).fit(q=.5); print('median reg',q.params['mid'],q.pvalues['mid'])
rng=np.random.default_rng(1); a=s.Q132a[s.mid==0].values; b=s.Q132a[s.mid==1].values
bs=[np.median(rng.choice(b,len(b)))-np.median(rng.choice(a,len(a))) for _ in range(5000)]; print('median diff',np.median(b)-np.median(a),np.percentile(bs,[2.5,97.5]))
print('mean diff',b.mean()-a.mean(), stats.ttest_ind(b,a,equal_var=False).pvalue)
# missingness
d['miss']=d.Q132a.isna().astype(int); print(d.groupby('survey').miss.mean()); print(smf.logit('miss~mid',d).fit(disp=0).pvalues['mid'])
print('trimmed mean', stats.trim_mean(a,.1),stats.trim_mean(b,.1))
# cluster-level paired
vv=d.groupby(['village','survey']).agg(own=('animal_own','mean'),att=('reg_att','mean')).unstack()
for k in ['own','att']: print(k,'villages up',(vv[k]['midline']>vv[k]['baseline']).sum(),'/10', stats.ttest_rel(vv[k]['midline'],vv[k]['baseline']).pvalue)
