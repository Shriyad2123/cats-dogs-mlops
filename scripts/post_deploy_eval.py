import argparse, json
from pathlib import Path
import requests

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--url",required=True); ap.add_argument("--test-dir",default="data/processed/test"); args=ap.parse_args()
    rows=[]
    for label in ["cat","dog"]:
        for p in list((Path(args.test_dir)/label).glob("*.jpg"))[:20]:
            with open(p,"rb") as f:
                r=requests.post(args.url.rstrip("/")+"/predict",files={"file":(p.name,f,"image/jpeg")},timeout=30); r.raise_for_status()
            pred=r.json()["label"]; rows.append((label,pred))
    acc=sum(a==b for a,b in rows)/len(rows)
    out={"samples":len(rows),"accuracy":acc}
    Path("reports/post_deploy_metrics.json").write_text(json.dumps(out,indent=2)); print(out)
if __name__=="__main__": main()
