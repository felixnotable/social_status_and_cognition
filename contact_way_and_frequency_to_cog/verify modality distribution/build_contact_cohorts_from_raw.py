#!/usr/bin/env python3
"""Build complete contact cohorts directly from original HRS source releases.

No prior derived CSV is needed. Requirements: Python 3.10+, numpy, pandas.
Use --help for the required original files and command syntax.
Optional --compare-csv is validation-only and never supplies output values.
"""
import argparse
import hashlib
import json
import platform
import re
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

TIMING={'A':(2006,2010,2012,2014),'B':(2008,2012,2014,2016)}
RELATIONS=('children','other_relatives','friends')
MODES=('inperson','phone','written_email')
TP=('T1','T2','T3','T4')
EARLY={'children':('LB007','LB009'),'other_relatives':('LB011','LB013'),'friends':('LB015','LB017')}
LATE={'children':('LB006','LB008'),'other_relatives':('LB010','LB012'),'friends':('LB014','LB016')}
SEX={1:'Male',2:'Female'}
RACE={1:'White, non-Hispanic',2:'Black/African American, non-Hispanic',3:'Other, non-Hispanic'}
MARITAL={1:'Married',2:'Married, spouse absent',3:'Partnered',4:'Separated',5:'Divorced',
         6:'Separated/Divorced',7:'Widowed',8:'Never married'}
FLAGS=('fimrc_imp','fdlrc_imp','fser7_imp','fbwc20_imp')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as src:
        for block in iter(lambda:src.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def unique(df,keys,name):
    if df.duplicated(keys).any():raise ValueError(f'Duplicate keys in {name}: {keys}')

def normalized_name(s):
    # Permit browser download suffixes such as H14LB_R(1).da.
    return re.sub(r'\(\d+\)(?=\.)','',Path(s).name).lower()

def resolve_source(location, basename, work):
    """Accept a file, source zip, or directory; extract only a named member.

    Never extract untrusted paths from a zip. Multiple copies must be identical.
    Input zip containers remain untouched. Empty/missing/ambiguous files fail.
    """
    location=Path(location)
    if not location.exists():raise FileNotFoundError(location)
    candidates=[]; target=normalized_name(basename)
    files=list(location.rglob('*')) if location.is_dir() else [location]
    direct=[p for p in files if p.is_file() and normalized_name(p.name)==target]
    if direct:
        if len({sha256(p) for p in direct})>1:raise ValueError(f'Conflicting copies of {basename}')
        return sorted(direct)[0]
    for p in files:
        if p.is_file() and p.suffix.lower()=='.zip':
            with zipfile.ZipFile(p) as z:
                for member in z.namelist():
                    if normalized_name(member)==target:candidates.append((p,member))
    if not candidates:raise FileNotFoundError(f'{basename} not found in {location}')
    work.mkdir(parents=True,exist_ok=True); dest=work/basename
    hashes=set()
    for i,(p,member) in enumerate(candidates):
        with zipfile.ZipFile(p) as z:
            if i==0:
                with z.open(member) as src, dest.open('wb') as out:shutil.copyfileobj(src,out)
                hashes.add(sha256(dest))
            else:
                with z.open(member) as src:hashes.add(hashlib.sha256(src.read()).hexdigest())
    if len(hashes)>1:raise ValueError(f'Conflicting archived copies of {basename}')
    return dest

def numeric_ids(series,name):
    x=pd.to_numeric(series,errors='raise')
    if x.isna().any() or (x%1!=0).any():raise ValueError(f'Invalid ID in {name}')
    return x.astype('int64')

def extract_lb(da,dct,year):
    prefix={2006:'K',2008:'L',2010:'M',2012:'N',2014:'O',2016:'P'}[year]
    cats=EARLY if year<=2012 else LATE
    wanted=['HHID','PN']+[prefix+x for x in ('LBRTYPE','LBELIG','LBCOMP')]
    for has,stem in cats.values():wanted += [prefix+has]+[prefix+stem+letter for letter in 'ABC']
    specs={}
    for line in dct.read_text(errors='replace').splitlines():
        m=re.search(r'_column\((\d+)\)\s+\w+\s+(\w+)\s+%(\d+)',line,re.I)
        if m and m[2].upper() in wanted:
            start=int(m[1])-1;specs[m[2].upper()]=(start,start+int(m[3]))
    if set(wanted)-set(specs):raise ValueError(f'Missing dictionary variables: {set(wanted)-set(specs)}')
    # DCT byte offsets are zero-based here. Only selected fields are read.
    raw=pd.read_fwf(da,colspecs=[specs[k] for k in wanted],names=wanted,header=None,dtype=str)
    raw=raw.apply(pd.to_numeric,errors='coerce')
    ids=numeric_ids(raw.HHID,'LB HHID')*1000+numeric_ids(raw.PN,'LB PN')
    out=pd.DataFrame({'hhidpn_key':ids,'year':year,'wave':(year-1990)//2,
        'lb_raw_record':True,'lb_completion':raw[prefix+'LBCOMP'],
        'lb_eligible':raw[prefix+'LBELIG'],'lb_respondent_type':raw[prefix+'LBRTYPE']})
    for rel,(has,stem) in cats.items():
        group=raw[prefix+has];out[rel+'_has_group_raw']=group
        for mode,letter in zip(MODES,'ABC'):
            v=raw[prefix+stem+letter];out[f'{rel}_{mode}_raw']=v
            score=(6-v).where(v.isin([1,2,3,4,5,6]))
            # Preserve the analysis conventions, including explicit group absence.
            score.loc[group.eq(5)]=np.nan if rel=='children' else 0.
            out[f'{rel}_{mode}_freqscore']=score
    out['lb_contact_screen_observed']=out[[r+'_has_group_raw' for r in RELATIONS]].notna().any(axis=1)
    unique(out,['hhidpn_key'],'LB '+str(year))
    return out

def load_stata(path,columns):
    # Public pandas API converts all Stata missing codes to NaN, but not labels.
    with pd.read_stata(path,columns=columns,convert_categoricals=False,chunksize=1000) as reader:
        return pd.concat(reader,ignore_index=True)

def compare_reference(new,path,outdir):
    old=pd.read_csv(path,low_memory=False).set_index('hhidpn_key');new=new.set_index('hhidpn_key')
    unique(old.reset_index(),['hhidpn_key'],'comparison CSV')
    if set(new.index)!=set(old.index):raise ValueError('Participant set differs from comparison CSV')
    old=old.reindex(new.index);rows=[]
    for c in sorted(set(new)&set(old)):
        a,b=new[c],old[c];both=a.notna()&b.notna()
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diff=(a.astype(float)-b.astype(float)).abs();mismatch=both&(diff>1e-10)
            substantive=both & ~np.isclose(a,b,rtol=5e-6,atol=1e-8,equal_nan=True)
            maxdiff=float(diff[both].max()) if both.any() else np.nan
        else:
            mismatch=both&a.astype(str).str.lower().ne(b.astype(str).str.lower())
            substantive=mismatch;maxdiff=np.nan
        rows.append({'column':c,'new_nonmissing':int(a.notna().sum()),'reference_nonmissing':int(b.notna().sum()),
            'gained':int((a.notna()&b.isna()).sum()),'lost':int((a.isna()&b.notna()).sum()),
            'nonidentical_nonmissing':int(mismatch.sum()),'substantive_value_mismatch':int(substantive.sum()),
            'max_absolute_difference':maxdiff})
    comparison=pd.DataFrame(rows);comparison.to_csv(outdir/'reference_comparison.csv',index=False)
    (outdir/'reference_schema_comparison.json').write_text(json.dumps({
        'new_only':sorted(set(new)-set(old)),'reference_only':sorted(set(old)-set(new))},indent=2))
    print('Comparison: '+str(len(comparison))+' common columns; details in reference_comparison.csv',flush=True)
    if comparison[['gained','lost','substantive_value_mismatch']].to_numpy().sum():
        raise ValueError('Reference differs in availability or values beyond rounding tolerance; inspect comparison report')

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lb-dir',required=True,type=Path,help='Original H06/H08/H10/H12/H14/H16LB_R .da + .dct files, or their zip archives')
    p.add_argument('--langa-weir',required=True,type=Path,help='cogfinalimp_9522wide.dta, its ZIP or containing directory')
    p.add_argument('--rand',required=True,type=Path,help='randhrs1992_2022v1.dta, its ZIP or containing directory')
    p.add_argument('--latent',required=True,type=Path,help='Dementia_HRS_2000-2016_Basic_Release1_2m.dta, its ZIP or directory')
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--compare-csv',type=Path,help='Optional prior CSV for validation only; NEVER a build input')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);work=out/'source_cache'
    sources=[];lbs=[]
    for year in range(2006,2017,2):
        da=resolve_source(args.lb_dir,f'H{year%100:02d}LB_R.da',work)
        dct=resolve_source(args.lb_dir,f'H{year%100:02d}LB_R.dct',work)
        sources += [da,dct];lbs.append(extract_lb(da,dct,year))
    lb=pd.concat(lbs,ignore_index=True);unique(lb,['hhidpn_key','year'],'LB person-wave')
    print(f'Extracted {len(lb):,} original LB person-wave records',flush=True)
    lpath=resolve_source(args.langa_weir,'cogfinalimp_9522wide.dta',work)
    rpath=resolve_source(args.rand,'randhrs1992_2022v1.dta',work)
    cpath=resolve_source(args.latent,'Dementia_HRS_2000-2016_Basic_Release1_2m.dta',work)
    sources += [lpath,rpath,cpath]
    lwcols=['hhid','pn']+[f'{v}{year}' for year in range(2006,2017,2) for v in
        ['cogtot27_imp','interview','proxy']+list(FLAGS)]
    lw=load_stata(lpath,lwcols)
    lw['hhidpn_key']=numeric_ids(lw.hhid,'LW HHID')*1000+numeric_ids(lw.pn,'LW PN')
    unique(lw,['hhidpn_key'],'LW');lw=lw.set_index('hhidpn_key')
    rcols=['hhidpn','ragender','raedyrs','raracem','rahispan']+[
        f'{prefix}{w}{suffix}' for w in range(8,14) for prefix,suffix in
        [('r','agey_e'),('r','mstat'),('h','atotw'),('r','iwstat')]]
    print('Reading original RAND source (wide file; may take several minutes)...',flush=True)
    rand=load_stata(rpath,rcols);rand['hhidpn_key']=numeric_ids(rand.hhidpn,'RAND HHIDPN')
    unique(rand,['hhidpn_key'],'RAND');rand=rand.set_index('hhidpn_key')
    cog=load_stata(cpath,['hhidpn','wave','Cog']);cog['hhidpn_key']=numeric_ids(cog.hhidpn,'latent HHIDPN')
    unique(cog,['hhidpn_key','wave'],'latent cognition')
    frames=[];overlap=[];coverage=[]
    for co,years in TIMING.items():
        baseline=lb[lb.year.eq(years[0])&lb.lb_contact_screen_observed].copy()
        if frames:
            dup=baseline.hhidpn_key.isin(frames[0].hhidpn_key)
            overlap=baseline.loc[dup,'hhidpn_key'].tolist();baseline=baseline.loc[~dup]
        wide=baseline[['hhidpn_key']].reset_index(drop=True);wide['cohort']=co;ids=wide.hhidpn_key
        for tp,y in zip(TP,years):
            wide['cognition_year_'+tp]=y
            wide['contact_year_'+tp]=years[3] if tp in ['T3','T4'] else y
        for tp,y in zip(TP,years):
            w=(y-1990)//2;cs=cog[cog.wave.eq(w)].set_index('hhidpn_key').Cog
            wide['Cog_'+tp]=ids.map(cs)
            score=ids.map(lw[f'cogtot27_imp{y}'])
            if not score.dropna().between(0,27).all():raise ValueError('Invalid original cognition score')
            wide['cog27_'+tp]=score
            observed=ids.map(rand[f'r{w}iwstat']).eq(1)
            wide['age_'+tp]=ids.map(rand[f'r{w}agey_e']).where(observed)
            wide['sex_'+tp]=ids.map(rand.ragender).map(SEX).where(observed)
            wide['education_'+tp]=ids.map(rand.raedyrs).where(observed)
            race=ids.map(rand.raracem).map(RACE);hisp=ids.map(rand.rahispan)
            race=race.where(hisp.eq(0));race.loc[hisp.eq(1)]='Hispanic'
            wide['race_ethnicity_'+tp]=race.where(observed)
            # Full source precision; no old six-significant-digit CSV rounding.
            wide['wealth_'+tp]=ids.map(rand[f'h{w}atotw']).where(observed)
            wide['marital_status_'+tp]=ids.map(rand[f'r{w}mstat']).map(MARITAL).where(observed)
            wide['interview_'+tp]=ids.map(lw[f'interview{y}'])
            wide['proxy_'+tp]=ids.map(lw[f'proxy{y}'])
            wide['core_record_'+tp]=observed
            coverage.append({'cohort':co,'timepoint':tp,'year':y,'baseline_N':len(wide),
                'cog27_N':int(score.notna().sum()),'latent_Cog_N':int(wide['Cog_'+tp].notna().sum()),
                'RAND_interview_N':int(observed.sum())})
        for tp,y in [('T1',years[0]),('T2',years[1]),('T3',years[3]),('T4',years[3])]:
            sub=lb[lb.year.eq(y)].drop(columns=['year','wave'])
            wide=wide.merge(sub.rename(columns={c:c+'_'+tp for c in sub if c!='hhidpn_key'}),
                on='hhidpn_key',how='left',validate='one_to_one')
        for rel in RELATIONS:
            for mode in MODES:
                base=f'{rel}_{mode}_freqscore'
                wide[f'{rel}_{mode}_change_T2_minus_T1']=wide[base+'_T2']-wide[base+'_T1']
                for tp in ['T3','T4']:
                    wide[f'{rel}_{mode}_change_{tp}_minus_T2']=wide[base+'_'+tp]-wide[base+'_T2']
            for tp in TP:
                wide[f'{rel}_equal_weight_total_{tp}']=wide[[f'{rel}_{m}_freqscore_{tp}' for m in MODES]].mean(axis=1,skipna=False)
            wide[f'{rel}_equal_weight_change_T2_minus_T1']=wide[f'{rel}_equal_weight_total_T2']-wide[f'{rel}_equal_weight_total_T1']
            for tp in ['T3','T4']:
                wide[f'{rel}_equal_weight_change_{tp}_minus_T2']=wide[f'{rel}_equal_weight_total_{tp}']-wide[f'{rel}_equal_weight_total_T2']
        for tp,y in zip(TP,years):
            for flag in FLAGS:wide[flag+'_'+tp]=wide.hhidpn_key.map(lw[flag+str(y)])
        frames.append(wide)
    data=pd.concat(frames,ignore_index=True).copy();unique(data,['hhidpn_key'],'final cohort')
    # These exact convenience variables allow the existing screening model to
    # consume this final CSV directly; they do not affect cohort inclusion.
    for mode in MODES:
        data['base_'+mode]=data[f'friends_{mode}_freqscore_T1']
        data['change_'+mode]=data[f'friends_{mode}_freqscore_T2']-data[f'friends_{mode}_freqscore_T1']
    data['meeting_loss']=np.where(data.change_inperson.notna(),data.change_inperson.lt(0).astype(float),np.nan)
    data['age75']=data.age_T2-75;data['wealth_ihs']=np.arcsinh(data.wealth_T2)
    data['marital_model']=data.marital_status_T2.replace({'Separated':'Separated/Divorced','Divorced':'Separated/Divorced'})
    for rel in RELATIONS:
        for mode in MODES:
            base=f'{rel}_{mode}_freqscore'
            pd.testing.assert_series_equal(data[base+'_T3'],data[base+'_T4'],check_names=False)
    if any('loneliness' in c or c.startswith('eligible_') for c in data):raise ValueError('Unexpected old eligibility column')
    data.to_csv(out/'contact_cohorts_A_B_no_loneliness_filter.csv',index=False)
    lb.to_csv(out/'lb_contact_modes_from_raw.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'source_coverage.csv',index=False)
    counts=[]
    cov=['age_T2','sex_T2','education_T2','race_ethnicity_T2','wealth_T2','marital_status_T2']
    for co in ['Pooled','A','B']:
        sub=data if co=='Pooled' else data[data.cohort.eq(co)]
        for outcome in ['T3','T4']:
            for rel in RELATIONS:
                cols=[f'{rel}_{m}_freqscore_{tp}' for m in MODES for tp in ['T1','T2']]
                cc=sub.dropna(subset=cols+cov+['cog27_T2','cog27_'+outcome])
                counts.append({'cohort':co,'outcome':outcome,'relationship':rel,'baseline_N':len(sub),'complete_N':len(cc)})
    pd.DataFrame(counts).to_csv(out/'sample_counts.csv',index=False)
    # Each column has a machine-readable source and meaning entry.
    dictionary=[]
    for col in data:
        source='Derived'
        if col.startswith('Cog_'):source='Dementia_HRS: Cog by HHIDPN and wave'
        elif col.startswith(('cog27_','interview_','proxy_')+FLAGS):source='Langa-Weir original source by HHIDPN and year'
        elif col.startswith(tuple(x+'_' for x in ['age','sex','education','race_ethnicity','wealth','marital_status','core_record'])):source='RAND source; covariates present only at interviewed waves'
        elif any(s in col for s in ['_raw_','_freqscore_']) or col.startswith('lb_'):source='Original LB .da/.dct; freqscore=6−valid raw score; group-absence rules in README'
        meaning=col.replace('_',' ')
        if col.startswith('contact_year_T3'):meaning='Legacy third LB year; same as contact T4, NOT cognition T3'
        if col.startswith('wealth_T'):meaning='RAND HwATOTW: wealth excluding IRA, dollars, full source precision'
        if col.startswith('core_record_'):meaning='RAND RwIWSTAT=1, not membership in an inherited derived panel'
        dictionary.append({'column':col,'source':source,'meaning':meaning})
    pd.DataFrame(dictionary).to_csv(out/'column_dictionary.csv',index=False)
    manifest={'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
        'builder_sha256':sha256(Path(__file__)),
        'inputs':[{'filename':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sources],
        'cohort_N':data.groupby('cohort').size().to_dict(),'total_N':len(data),'columns':len(data.columns),
        'double_baseline_assigned_to_A':overlap,'raw_build_uses_derived_csv':False,
        'historical_audit_columns_omitted':['cog27_restored_T1–T4','inherited_core_record_T1–T4']}
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.compare_csv:compare_reference(data,args.compare_csv,out)
    print(f'Wrote {len(data):,} people × {len(data.columns)} columns. Cohorts: {manifest["cohort_N"]}',flush=True)
    print(pd.DataFrame(counts).query("relationship=='friends'").to_string(index=False),flush=True)

if __name__=='__main__':main()
