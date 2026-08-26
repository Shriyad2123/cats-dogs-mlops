import argparse, time, requests

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--image", required=True); args=ap.parse_args()
    base=args.url.rstrip("/")
    last=None
    for _ in range(20):
        try:
            r=requests.get(base+"/health", timeout=20); last=r
            if r.ok: break
        except Exception as e: last=e
        time.sleep(6)
    if not getattr(last, "ok", False): raise SystemExit(f"health failed: {last}")
    with open(args.image,"rb") as f:
        r=requests.post(base+"/predict", files={"file": ("sample.jpg",f,"image/jpeg")}, timeout=30)
    r.raise_for_status(); print("health:", requests.get(base+"/health").json()); print("prediction:",r.json())

if __name__ == "__main__": main()
