"""Streamlit dashboard for the SIH26184 research prototype."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.preprocess import preprocess_complaints
from src.features import FeatureBuilder
from src.utils import HOTSPOTS

st.set_page_config(page_title="SIH26184 | Cash-out Forecasting",layout="wide")
st.title("Cybercrime Pattern Grouping & Cash-Withdrawal Forecasting")
st.caption("Smart India Hackathon · Mapping ID SIH26184 · Synthetic-data research prototype")
processed=ROOT/"data/processed"; model_path=ROOT/"models/pipeline.joblib"
def load_csv(name):
    p=processed/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
page=st.sidebar.radio("Navigation",["Overview","Cluster Explorer","Predict","Validation","Upload"])
if page=="Overview":
    c=load_csv("complaints_clean.csv"); w=load_csv("withdrawals_clean.csv")
    if c.empty: st.info("Run `python main.py` first to generate the demonstration dataset.")
    else:
        a,b,d=st.columns(3); a.metric("Complaints",f"{len(c):,}"); b.metric("Confirmed withdrawals",f"{len(w):,}"); d.metric("Fraud categories",c.fraud_type.nunique())
        st.subheader("Complaint mix"); st.bar_chart(c.fraud_type.value_counts())
        st.subheader("Victim state distribution"); st.bar_chart(c.victim_state.value_counts().head(12))
        st.dataframe(c.head(20),use_container_width=True)
elif page=="Cluster Explorer":
    c=load_csv("complaints_clean.csv"); w=load_csv("withdrawals_clean.csv")
    if c.empty: st.warning("Run main.py first.")
    else:
        metric=st.selectbox("Similarity metric",["euclidean","cosine"]); algo=st.selectbox("Algorithm",["kmeans","agglomerative","dbscan"])
        if st.button("Build cluster view"):
            from src.clustering import fit_cluster
            fb=FeatureBuilder(); X=fb.fit_transform(c)["combined"]
            # Agglomerative / DBSCAN use a bounded sample for browser responsiveness.
            n=min(1800,len(c)); Xs=X[:n]
            _,lab=fit_cluster(Xs,metric,algo,n_clusters=6)
            view=c.iloc[:n].copy(); view["cluster"]=lab
            m=folium.Map(location=[22.5,79],zoom_start=5)
            palette=["red","blue","green","purple","orange","darkred","cadetblue","darkgreen"]
            for _,r in view.iterrows():
                colour=palette[int(r.cluster)%len(palette)] if r.cluster>=0 else "gray"
                folium.CircleMarker([r.victim_lat,r.victim_lon],radius=3,popup=f"{r.complaint_id} · cluster {r.cluster}",color=colour,fill=True,fill_opacity=.65).add_to(m)
            st_folium(m,width=1000,height=550)
            st.bar_chart(view.cluster.value_counts())
elif page=="Predict":
    if not model_path.exists(): st.warning("Run `python main.py` first.")
    else:
        model=joblib.load(model_path); fb=model["feature_builder"]
        with st.form("new_case"):
            city=st.text_input("Victim city","Ludhiana"); state=st.text_input("Victim state","Punjab")
            lat=st.number_input("Victim latitude",value=30.9010); lon=st.number_input("Victim longitude",value=75.8573)
            chosen_date=st.date_input("Complaint date",value=pd.Timestamp.now().date())
            chosen_time=st.time_input("Complaint time",value=pd.Timestamp.now().time().replace(second=0,microsecond=0))
            dt=pd.Timestamp.combine(chosen_date,chosen_time)
            ft=st.selectbox("Fraud type",["UPI fraud","OTP scam","fake KYC","job scam","investment scam","sextortion","phishing"])
            mo=st.text_area("Modus-operandi description","QR code collect request")
            amount=st.number_input("Amount lost (INR)",min_value=0.0,value=10000.0); bank=st.selectbox("Bank",["SBI","HDFC","ICICI","Axis","PNB","Kotak","Other"])
            channel=st.selectbox("Transaction channel",["UPI","IMPS","NEFT","Card","Wallet","NetBanking"]); hops=st.number_input("Number of mule-account hops",0,20,2)
            metric=st.selectbox("Forecast distance metric",["euclidean","cosine"])
            go=st.form_submit_button("Forecast likely withdrawal locations")
        if go:
            row=pd.DataFrame([{"complaint_id":"NEW","complaint_datetime":dt,"victim_city":city,"victim_state":state,"victim_lat":lat,"victim_lon":lon,"fraud_type":ft,"mo_description":mo,"amount_lost":amount,"bank_name":bank,"transaction_channel":channel,"number_of_hops":hops}])
            x=fb.transform(row)["combined"]; result=model["forecasters"][metric].predict(x,5)
            st.subheader("Top predicted cash-out locations")
            st.dataframe(pd.DataFrame(result["predictions"]),use_container_width=True)
            st.info(result["time_window"])
            mp=folium.Map(location=[lat,lon],zoom_start=5)
            for p in result["predictions"]:
                if p["location"] in HOTSPOTS:
                    h=HOTSPOTS[p["location"]]; folium.Marker([h[0],h[1]],popup=f'{p["location"]}: {p["probability"]:.1%}').add_to(mp)
            st_folium(mp,width=900,height=450)
elif page=="Validation":
    v=load_csv("validation_metrics.csv"); a=load_csv("ablation.csv"); comp=load_csv("algorithm_comparison.csv")
    if v.empty: st.warning("Run main.py first.")
    else:
        st.subheader("Euclidean vs Cosine — temporal holdout")
        st.dataframe(v,use_container_width=True); st.bar_chart(v.set_index("metric")[["top1_hit_rate","top3_hit_rate","top5_hit_rate"]])
        st.subheader("Algorithm comparison"); st.dataframe(comp,use_container_width=True)
        st.subheader("Feature ablation"); st.dataframe(a,use_container_width=True)
        p=processed/"conclusion.txt"
        if p.exists(): st.success(p.read_text())
        st.caption("ARI/NMI are reported as reference-label diagnostics; interpret cautiously for single-location-per-case synthetic labels.")
elif page=="Upload":
    st.write("Upload a complaints CSV with the documented complaint schema. Confirmed withdrawal labels are needed for supervised validation.")
    up=st.file_uploader("Complaints CSV",type=["csv"])
    if up:
        try:
            c=preprocess_complaints(pd.read_csv(up)); st.success(f"Loaded {len(c)} valid unique complaints.")
            st.dataframe(c.head(30),use_container_width=True)
            if st.button("Download cleaned complaints"): st.download_button("Download CSV",c.to_csv(index=False),"cleaned_complaints.csv","text/csv")
        except Exception as e: st.error(str(e))
    for name in ["validation_metrics.csv","algorithm_comparison.csv","ablation.csv","complaints_clean.csv","withdrawals_clean.csv"]:
        p=processed/name
        if p.exists(): st.download_button(f"Download {name}",p.read_bytes(),name,"text/csv")
    if (processed/"summary.json").exists():
        st.download_button("Download pipeline summary JSON",(processed/"summary.json").read_bytes(),"summary.json","application/json")
    st.caption("PDF report: generate from the CLI with `python report.py`.")
