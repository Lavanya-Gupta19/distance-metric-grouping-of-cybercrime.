"""Run the end-to-end synthetic pipeline from the project root."""
from __future__ import annotations
import logging, json
from pathlib import Path
import joblib, pandas as pd
from sklearn.cluster import KMeans
from src.generate_data import generate_data
from src.preprocess import preprocess_complaints,preprocess_withdrawals,save_sqlite
from src.features import FeatureBuilder
from src.clustering import compare_algorithms,fit_cluster,evaluate_clusters
from src.forecast import LocationForecaster
from src.validate import run_validation
from src.utils import ROOT,setup_logging,SEED

def main():
    setup_logging(); log=logging.getLogger("pipeline")
    raw=ROOT/"data/raw"; processed=ROOT/"data/processed"; models=ROOT/"models"
    cpath=raw/"complaints.csv"; wpath=raw/"confirmed_withdrawals.csv"
    if cpath.exists() and wpath.exists():
        c=pd.read_csv(cpath); w=pd.read_csv(wpath)
    else: c,w=generate_data(10000,out_dir=raw)
    c=preprocess_complaints(c); w=preprocess_withdrawals(w)
    # one withdrawal record per complaint; join alignment is deterministic.
    w=w.set_index("complaint_id").loc[c.complaint_id].reset_index()
    save_sqlite(c,w,ROOT/"data/cybercrime.sqlite")
    c.to_csv(processed/"complaints_clean.csv",index=False); w.to_csv(processed/"withdrawals_clean.csv",index=False)
    fb=FeatureBuilder(); groups=fb.fit_transform(c); X=groups["combined"]
    log.info("Built feature matrix: %s",X.shape)
    comparison,idx,labels=compare_algorithms(X)
    comparison.to_csv(processed/"algorithm_comparison.csv",index=False)
    # 2D projection and cluster-size distribution for a bounded representative sample.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD
    proj=TruncatedSVD(n_components=2,random_state=SEED).fit_transform(X[idx])
    sample_labels=labels.get("euclidean_kmeans")
    fig,ax=plt.subplots(figsize=(8,5))
    scatter=ax.scatter(proj[:,0],proj[:,1],c=sample_labels,s=8,cmap="tab10",alpha=.65)
    ax.set_title("Complaint feature projection (TruncatedSVD, Euclidean KMeans)")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    fig.colorbar(scatter,ax=ax,label="Cluster"); fig.tight_layout()
    fig.savefig(processed/"cluster_projection.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    pd.Series(sample_labels).value_counts().sort_index().plot(kind="bar",ax=ax)
    ax.set_title("Cluster size distribution"); ax.set_xlabel("Cluster"); ax.set_ylabel("Complaints")
    fig.tight_layout(); fig.savefig(processed/"cluster_sizes.png",dpi=150); plt.close(fig)
    # Deployable forecasting model uses stable KMeans on all training records.
    _,full_labels=fit_cluster(X,"euclidean","kmeans",n_clusters=6)
    forecasters={}
    for metric in ["euclidean","cosine"]:
        _,lab=fit_cluster(X,metric,"kmeans",n_clusters=6)
        forecasters[metric]=LocationForecaster(metric).fit(X,lab,w)
    validation=run_validation(c,w,FeatureBuilder(),n_clusters=6,max_test=350)
    validation.to_csv(processed/"validation_metrics.csv",index=False)
    # Ablation with same fixed KMeans setup and silhouette diagnostics.
    ab=[]
    for key in ["geo_only","temporal_only","mo_only","combined"]:
        xx=groups[key]
        for metric in ["euclidean","cosine"]:
            _,lab=fit_cluster(xx,metric,"kmeans",n_clusters=6)
            ev=evaluate_clusters(xx,lab,metric)
            ab.append({"feature_set":key,"metric":metric,**ev})
    pd.DataFrame(ab).to_csv(processed/"ablation.csv",index=False)
    joblib.dump({"feature_builder":fb,"forecasters":forecasters,"labels":full_labels},models/"pipeline.joblib")
    summary={"complaints":len(c),"withdrawals":len(w),"features":X.shape[1],
             "algorithm_comparison":comparison.replace({float("nan"):None}).to_dict("records"),
             "validation":validation.round(4).to_dict("records")}
    (processed/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    log.info("Validation results:\\n%s",validation.to_string(index=False))
    if len(validation):
        best=validation.sort_values("top5_hit_rate",ascending=False).iloc[0]
        other=validation[validation.metric!=best.metric].iloc[0]
        conclusion=f"{best.metric.title()} has the higher Top-5 hit rate ({best.top5_hit_rate:.3f} vs {other.top5_hit_rate:.3f}); treat the difference as indicative only because this is synthetic data."
        (processed/"conclusion.txt").write_text(conclusion)
        print("\\nFINAL EUCLIDEAN VS COSINE SUMMARY\\n"+validation.to_string(index=False)+"\\n"+conclusion)
    return summary
if __name__=="__main__": main()
