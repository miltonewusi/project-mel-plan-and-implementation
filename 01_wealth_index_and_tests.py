import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

F='ghana_me_dataset.xlsx'
d=pd.read_excel(F,sheet_name='dataset')
d['mid']=(d.survey=='midline').astype(int)

# (a),(b) check
d['animal_own']=(d[['Q033','Q034','Q035','Q036']]>0).any(axis=1).astype(int)
d['reg_att']=d.Q121.isin([3,4]).astype(int)

# (c) descriptives
print(d.groupby('survey')[['animal_own','reg_att']].agg(['count','mean','std']))
print(d.groupby('survey').Q132a.describe())

# (d) wealth index
bin_vars=['Q022','Q023','Q024','Q025','Q026','Q027','Q028','Q029','Q030','Q031','Q032']
cnt_vars=['Q033','Q034','Q035','Q036','Q037','Q038','Q039','Q040']
X=pd.DataFrame(index=d.index)
for v in bin_vars: X[v]=(d[v]==1).astype(int)
for v in cnt_vars: X[v]=np.log1p(d[v])
sav=d.Q132a.copy()
pooled_med=sav.median()
X['sav']=np.log1p(sav.fillna(pooled_med))
def pca_index(X,mask=None):
    Z=X if mask is None else X[mask]
    sc=StandardScaler().fit(Z); p=PCA(n_components=3).fit(sc.transform(Z))
    s=sc.transform(X)@p.components_[0]
    if p.components_[0][list(X.columns).index('Q032')]<0: s=-s; load=-p.components_[0]
    else: load=p.components_[0]
    return s,pd.Series(load,X.columns),p.explained_variance_ratio_
d['wealth'],load,evr=pca_index(X)
print(evr); print(load.round(3))
print(d.groupby('survey').wealth.agg(['mean','std','median']))
t=stats.ttest_ind(d.wealth[d.mid==1],d.wealth[d.mid==0],equal_var=False); print('t',t)
print(stats.mannwhitneyu(d.wealth[d.mid==1],d.wealth[d.mid==0]))
m=smf.ols('wealth~mid',d).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.village)[0]}); print(m.params,m.pvalues)
m2=smf.ols('wealth~mid+C(village)',d).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.village)[0]}); print(m2.params['mid'],m2.pvalues['mid'])

def rich_tests(thr_col,label):
    r=d[thr_col]
    t=pd.crosstab(d.survey,r); 
    p1=r[d.mid==0].mean(); p2=r[d.mid==1].mean()
    chi=stats.chi2_contingency(t,correction=False)
    print(label,round(p1,3),round(p2,3),'chi p',chi[1])
    m=smf.logit(f'{thr_col}~mid+C(village)',d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.village)[0]})
    print('  FE logit OR',np.exp(m.params['mid']),m.pvalues['mid'])
    lpm=smf.ols(f'{thr_col}~mid+C(village)',d).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.village)[0]})
    print('  FE LPM',lpm.params['mid'],lpm.pvalues['mid'])
for q,name in [(0.8,'top20'),(0.6,'top40'),(0.5,'top50')]:
    d['rich_'+name]=(d.wealth>=d.wealth.quantile(q)).astype(int)
    rich_tests('rich_'+name,name+' pooled')
b80=d.wealth[d.mid==0].quantile(.8); d['rich_b80']=(d.wealth>=b80).astype(int); rich_tests('rich_b80','baseline-80th')
# per-village
rows=[]
for v,g in d.groupby('village'):
    b=g[g.mid==0].rich_top20; mm=g[g.mid==1].rich_top20
    tab=[[b.sum(),len(b)-b.sum()],[mm.sum(),len(mm)-mm.sum()]]
    rows.append((v,len(b),len(mm),b.mean(),mm.mean(),stats.fisher_exact(tab)[1]))
print(pd.DataFrame(rows,columns=['v','nb','nm','pb','pm','p']).round(3))
