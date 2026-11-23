"""Create a compact PDF report from pipeline outputs."""
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
ROOT=Path(__file__).resolve().parent
def create_report():
    out=ROOT/"data/processed/SIH26184_report.pdf"; styles=getSampleStyleSheet()
    story=[Paragraph("SIH26184 — Cash Withdrawal Forecasting",styles["Title"]),
           Paragraph("Synthetic-data research prototype; not an operational law-enforcement tool.",styles["Normal"]),Spacer(1,16)]
    for fn,title in [("validation_metrics.csv","Temporal holdout validation"),("algorithm_comparison.csv","Algorithm comparison"),("ablation.csv","Feature ablation")]:
        p=ROOT/"data/processed"/fn
        if p.exists():
            df=pd.read_csv(p).round(4); story.append(Paragraph(title,styles["Heading2"]))
            data=[list(df.columns)]+df.astype(str).values.tolist()
            t=Table(data,repeatRows=1); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#204f78")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)])); story.extend([t,Spacer(1,12)])
    story.append(Paragraph("Privacy & ethics: use anonymized, lawfully obtained records; human review and due process are mandatory. Predictions are leads for analysis only, never evidence of guilt.",styles["Normal"]))
    SimpleDocTemplate(str(out),pagesize=A4).build(story); return out
if __name__=="__main__": print(create_report())
