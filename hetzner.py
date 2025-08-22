#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch Hetzner Cloud servers to run tokenizer shards against data in S3.

High-level flow:
  1) Enumerate input S3 prefix -> size-aware greedy split into N shards (jsonl.gz manifests)
  2) Upload per-shard manifests to S3
  3) Create N Hetzner servers with cloud-init user-data that:
       - installs uv/python/awscli
       - downloads the code artifact zip from S3
       - downloads its shard manifest and fetches the shard’s inputs from S3
       - runs tokenizer.py with your exact CLI
       - syncs outputs to S3 and writes an S3 done marker
  4) Poll for all done markers, then delete the servers

Security note: workers need S3 credentials. Prefer passing short‑lived STS creds
via --assume-role-arn (embedded into user-data) rather than long‑lived keys.
"""

from __future__ import annotations
import argparse, gzip, io, json, os, random, string, sys, time
from urllib.parse import urlparse
from typing import List, Tuple

import boto3
import botocore

# pip install hcloud
from hcloud import Client as HClient

# --------------------
# Helpers (S3 & utils)
# --------------------
def parse_s3_uri(u: str) -> Tuple[str, str]:
    x = urlparse(u)
    assert x.scheme == "s3" and x.netloc, f"Bad S3 URI: {u}"
    return x.netloc, x.path.lstrip("/")

def s3_list_objects(s3, bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/"):
                continue
            yield obj["Key"], int(obj["Size"])

def greedy_balance(objects: List[Tuple[str, int]], n_bins: int):
    """
    objects: list of (key, size)
    returns: list[b] -> list[(key,size)] approximately balanced by total size
    """
    bins = [dict(size=0, items=[]) for _ in range(n_bins)]
    for k, sz in sorted(objects, key=lambda x: x[1], reverse=True):
        bins.sort(key=lambda b: b["size"])
        bins[0]["items"].append((k, sz)); bins[0]["size"] += sz
    return [b["items"] for b in bins]

def put_gzip_jsonl(s3, bucket: str, key: str, rows: List[dict]):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in rows:
            gz.write(json.dumps(r, separators=(",",":")).encode("utf-8"))
            gz.write(b"\n")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="application/json", ContentEncoding="gzip")

def wait_for_done_markers(s3, done_uris: List[str], poll_s=30, timeout_s=None):
    start = time.time()
    remaining = set(done_uris)
    while remaining:
        for uri in list(remaining):
            b, k = parse_s3_uri(uri)
            try:
                s3.head_object(Bucket=b, Key=k)
                remaining.remove(uri)
            except botocore.exceptions.ClientError as e:
                code = e.response["Error"]["Code"]
                if code not in ("404","NoSuchKey","NotFound"):
                    raise
        if not remaining:
            return True
        if timeout_s and (time.time() - start) > timeout_s:
            return False
        time.sleep(poll_s)
    return True

def rand_suffix(n=5):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

# --------------------------------------------
# Build cloud-init user-data for each server
# --------------------------------------------
def build_user_data_for_hetzner(
    aws_region: str,
    shard_index: int,
    manifest_s3_uri: str,
    input_prefix_s3: str,   # for logging
    output_prefix_s3: str,
    code_zip_s3_uri: str,
    tokenizer_cli_args: str,
    aws_creds: dict,        # {aws_access_key_id, aws_secret_access_key, aws_session_token}
    venv_dir="/opt/worker/venv",
    work_dir="/opt/worker",
    max_dl_workers=32,
):
    tok_args_json = json.dumps(tokenizer_cli_args or "")
    shard_name = f"shard-{shard_index:03d}"

    # NOTE: This closely mirrors your original EC2 user-data: install deps,
    # fetch code & manifest from S3, run tokenizer.py, sync outputs, write done marker
    # (preserving CLI shape and steps).
    # It additionally *injects AWS credentials* so S3 works on Hetzner.
    return f"""#cloud-config
runcmd:
  - 'bash -lc "set -euox pipefail; \
REGION={aws_region} \
MANIFEST_URI={manifest_s3_uri} \
INPUT_S3_PREFIX={input_prefix_s3.rstrip('/')} \
OUTPUT_S3_PREFIX={output_prefix_s3.rstrip('/')} \
CODE_ZIP_URI={code_zip_s3_uri} \
SHARD={shard_name} \
WORK_DIR={work_dir} \
VENV_DIR={venv_dir} \
DATA_DIR=/mnt/data/$SHARD \
OUT_DIR=/mnt/out/$SHARD \
LOG_DIR=/var/log/tokenizer \
DL_WORKERS={max_dl_workers} \
AWS_ACCESS_KEY_ID={aws_creds.get('aws_access_key_id','')} \
AWS_SECRET_ACCESS_KEY={aws_creds.get('aws_secret_access_key','')} \
AWS_SESSION_TOKEN={aws_creds.get('aws_session_token','')} \
TOK_ARGS=$(python3 - <<PY; import json; print(json.loads({tok_args_json!r})); PY); \
mkdir -p $WORK_DIR $DATA_DIR $OUT_DIR $LOG_DIR; \
echo region=$REGION shard=$SHARD input=$INPUT_S3_PREFIX output=$OUTPUT_S3_PREFIX | tee -a $LOG_DIR/$SHARD.log; \
trap '' EXIT; \
if command -v apt-get >/dev/null 2>&1; then PKG=apt-get; elif command -v dnf >/dev/null 2>&1; then PKG=dnf; elif command -v yum >/dev/null 2>&1; then PKG=yum; else PKG=apt-get; fi; \
sudo $PKG -y update || true; \
sudo $PKG -y install curl unzip jq awscli git tar || true; \
curl -LsSf https://astral.sh/uv/install.sh | sh; \
export PATH=$HOME/.local/bin:$PATH; \
# Configure AWS creds (short-lived if provided by STS)
export AWS_DEFAULT_REGION=$REGION; \
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID; \
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY; \
aws configure set aws_session_token $AWS_SESSION_TOKEN; \
aws configure set default.region $REGION; \
# Make a Python 3.11 venv even if OS python is older
uv python install 3.11 || true; \
uv venv --python 3.11 $VENV_DIR; \
. $VENV_DIR/bin/activate; \
uv pip install -U pip wheel setuptools boto3 botocore; \
# Pull code artifact
aws s3 cp $CODE_ZIP_URI $WORK_DIR/code.zip; \
rm -rf $WORK_DIR/code; unzip -q -o $WORK_DIR/code.zip -d $WORK_DIR/code; \
tokenizer_dir=$WORK_DIR/code/tokenizer; \
# Install tokenizer deps
uv pip install -r $tokenizer_dir/requirements.txt; \
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -n 1048576; \
# Fetch manifest for this shard
aws s3 cp $MANIFEST_URI $WORK_DIR/manifest.jsonl.gz; \
# Write small helper to fetch objects listed in manifest (same as your EC2 path) \
cat > $WORK_DIR/fetch_and_prepare.py <<PY2; \
import concurrent.futures, gzip, json, os, subprocess, sys; \
from urllib.parse import urlparse; \
import boto3; \
manifest_gz, extract_root, max_workers = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 16; \
def parse_s3(u): x=urlparse(u); assert x.scheme==\\\"s3\\\" and x.netloc; return x.netloc, x.path.lstrip(\\\"/\\\"); \
def rows(): \
  with gzip.open(manifest_gz, \\\"rt\\\") as f: \
    for line in f: j=json.loads(line); yield j[\\\"s3_uri\\\"]; \
s3=boto3.client(\\\"s3\\\"); os.makedirs(extract_root, exist_ok=True); \
def is_tar_gz(k): return k.endswith(\\\".tar.gz\\\"); \
def dl(b,k,d): os.makedirs(os.path.dirname(d), exist_ok=True); s3.download_file(b,k,d); \
def extract(arc,d): os.makedirs(d, exist_ok=True); subprocess.check_call([\\\"tar\\\",\\\"-xzf\\\",arc,\\\"-C\\\",d]); \
def handle(u): \
  b,k=parse_s3(u); \
  if is_tar_gz(k): \
    scratch=os.path.join(extract_root, \\\"_archives\\\"); os.makedirs(scratch, exist_ok=True); \
    local=os.path.join(scratch, os.path.basename(k)); dl(b,k,local); extract(local, extract_root); \
    try: os.remove(local) \
    except OSError: pass; return (\\\"arc\\\",k) \
  else: \
    rel=k.split(\\\"/\\\",1)[1] if \\\"/\\\" in k else os.path.basename(k); dest=os.path.join(extract_root, rel); dl(b,k,dest); return (\\\"obj\\\",k); \
def main(): \
  items=list(rows()); print(f\\\"[prep] items={{{{len(items)}}}} into {{{{extract_root}}}}\\\"); ok=0; \
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex: \
    for i,_ in enumerate(ex.map(handle, items, chunksize=4),1): ok+=1; \
      if i%50==0: print(f\\\"[prep] {{{{i}}}}/{{{{len(items)}}}}\\\"); \
  print(f\\\"[prep] done: {{{{ok}}}}/{{{{len(items)}}}}\\\"); \
if __name__==\\\"__main__\\\": main() \
PY2; \
uv run $WORK_DIR/fetch_and_prepare.py $WORK_DIR/manifest.jsonl.gz $DATA_DIR $DL_WORKERS 2>&1 | tee -a $LOG_DIR/$SHARD.log; \
# Run tokenizer (exact CLI preserved) \
cd $tokenizer_dir; CORES=$(nproc || echo 16); W=$(( CORES>8 ? CORES-4 : CORES )); \
echo [run] tokenizer workers=$W | tee -a $LOG_DIR/$SHARD.log; \
set +e; \
python tokenizer.py --midi-root $DATA_DIR --out-dir $OUT_DIR --num-workers $W --log-level INFO $TOK_ARGS 2>&1 | tee -a $LOG_DIR/$SHARD.log; \
rc=$?; set -e; \
# Upload outputs regardless; then mark done \
aws s3 sync $OUT_DIR $OUTPUT_S3_PREFIX/$SHARD/ --no-progress || true; \
python - <<PY3; import json, os; d={{'shard': os.environ.get('SHARD','unknown'),'host': os.uname().nodename,'exit_code': int(os.environ.get('rc','0')) if os.environ.get('rc') else 0}}; open('/opt/worker/DONE.json','w').write(json.dumps(d)); PY3; \
aws s3 cp /opt/worker/DONE.json $OUTPUT_S3_PREFIX/_done/$SHARD.done || true; \
# Also copy logs to S3 for debugging \
aws s3 cp $LOG_DIR/$SHARD.log $OUTPUT_S3_PREFIX/_logs/$SHARD.log || true \
"'
"""

# --------------------------------------------
# STS: short-lived S3 credentials for workers
# --------------------------------------------
def mint_worker_s3_creds(
    boto_session: boto3.Session,
    aws_region: str,
    input_prefix_s3: str,
    output_prefix_s3: str,
    role_arn: str | None,
    duration_seconds: int,
):
    """
    If role_arn is set -> AssumeRole with a restricted inline policy limited to
    the input/output prefixes. Otherwise, pass through env creds (discouraged).
    """
    in_b, in_k = parse_s3_uri(input_prefix_s3)
    out_b, out_k = parse_s3_uri(output_prefix_s3)
    in_k = in_k.rstrip("/") + "/*"
    out_k = out_k.rstrip("/") + "/*"

    if role_arn:
        sts = boto_session.client("sts", region_name=aws_region)
        policy = {
            "Version":"2012-10-17",
            "Statement":[
                {"Effect":"Allow","Action":["s3:GetObject"],"Resource":[f"arn:aws:s3:::{in_b}/{in_k}"]},
                {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":[f"arn:aws:s3:::{in_b}"],
                 "Condition":{"StringLike":{"s3:prefix":[in_k]}}},
                {"Effect":"Allow","Action":["s3:PutObject"],"Resource":[f"arn:aws:s3:::{out_b}/{out_k}",
                                                                          f"arn:aws:s3:::{out_b}/{out_k.replace('/*','/_logs/*')}",
                                                                          f"arn:aws:s3:::{out_b}/{out_k.replace('/*','/_done/*')}" ]},
                {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":[f"arn:aws:s3:::{out_b}"],
                 "Condition":{"StringLike":{"s3:prefix":[out_k, out_k.replace('/*','/_logs/*'), out_k.replace('/*','/_done/*')]}}}
            ]
        }
        resp = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=f"tok-{rand_suffix()}",
            DurationSeconds=max(900, min(duration_seconds, 43200)),
            Policy=json.dumps(policy),
        )
        c = resp["Credentials"]
        return {
            "aws_access_key_id": c["AccessKeyId"],
            "aws_secret_access_key": c["SecretAccessKey"],
            "aws_session_token": c["SessionToken"],
        }

    # Fallback: pass through ambient creds (less secure)
    creds = boto_session.get_credentials()
    if not creds or not creds.access_key or not creds.secret_key:
        raise RuntimeError("No AWS credentials available; set AWS_* or use --assume-role-arn")
    frozen = creds.get_frozen_credentials()
    return {
        "aws_access_key_id": frozen.access_key,
        "aws_secret_access_key": frozen.secret_key,
        "aws_session_token": frozen.token or "",
    }

# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser(description="Launch Hetzner fleet and tokenize MIDI shards (S3-backed)")
    # Hetzner
    ap.add_argument("--hcloud-token", default=os.environ.get("HCLOUD_TOKEN"), help="Hetzner Cloud API token")
    ap.add_argument("--location", default="nbg1", help="Hetzner location name (e.g., nbg1, fsn1, hel1, ash)")
    ap.add_argument("--server-type", default="cax51", help="Hetzner server type (e.g., cpx51, cax41)")
    ap.add_argument("--image", default="ubuntu-22.04", help="Hetzner image (name or ID), e.g., ubuntu-22.04")
    ap.add_argument("--ssh-key-names", default="", help="Comma-separated Hetzner SSH key names to inject")
    ap.add_argument("--network-id", type=int, default=None, help="Optional: attach to a private Network ID")
    ap.add_argument("--num-servers", type=int, required=True)

    # S3 / Task
    ap.add_argument("--aws-region", required=True, help="AWS region (for S3 and STS)")
    ap.add_argument("--s3-input-prefix", required=True, help="s3://bucket/prefix where MIDIs (.mid or .tar.gz) live")
    ap.add_argument("--s3-output-prefix", required=True, help="s3://bucket/prefix to upload tokenized shards")
    ap.add_argument("--s3-code-zip", required=True, help="s3://bucket/path to code artifact zip containing tokenizer.py & requirements.txt")
    ap.add_argument("--tokenizer-args", default="", help="Extra CLI args passed to tokenizer.py")

    ap.add_argument("--manifest-prefix", default=None, help="s3://bucket/prefix to store manifests (default <output>/_manifests/)")
    ap.add_argument("--timeout-min", type=int, default=0, help="Global timeout for orchestration (0=none)")
    ap.add_argument("--dl-workers", type=int, default=32, help="Parallel S3 downloads per worker during fetch")

    # Credentials for workers
    ap.add_argument("--assume-role-arn", default=None, help="IAM role ARN to assume for worker S3 access (recommended)")
    ap.add_argument("--cred-ttl-min", type=int, default=240, help="STS session duration for workers (minutes)")

    args = ap.parse_args()
    if not args.hcloud_token:
        print("Provide --hcloud-token or set HCLOUD_TOKEN", file=sys.stderr)
        sys.exit(2)

    session = boto3.Session(region_name=args.aws_region)
    s3 = session.client("s3")

    in_b, in_k = parse_s3_uri(args.s3_input_prefix)
    out_b, out_k = parse_s3_uri(args.s3_output_prefix)

    manifest_prefix = args.manifest_prefix or f"s3://{out_b}/{out_k.rstrip('/')}/_manifests/"
    man_b, man_k = parse_s3_uri(manifest_prefix if manifest_prefix.endswith("/") else manifest_prefix + "/")

    # -------- 1) Build manifests --------
    print("[plan] Listing S3 input objects …")
    objects = list(s3_list_objects(s3, in_b, in_k))
    if not objects:
        print("No input objects found under", args.s3_input_prefix, file=sys.stderr)
        sys.exit(2)

    bins = greedy_balance(objects, args.num_servers)
    print("[plan] Shards:", [sum(sz for _, sz in b) for b in bins])

    manifest_uris = []
    for i, b in enumerate(bins):
        rows = [{"s3_uri": f"s3://{in_b}/{k}"} for (k, _sz) in b]
        key = f"{man_k.rstrip('/')}/manifest-{i:03d}.jsonl.gz"
        put_gzip_jsonl(s3, man_b, key, rows)
        manifest_uris.append(f"s3://{man_b}/{key}")

    # Clear existing done markers
    done_uris = [f"s3://{out_b}/{out_k.rstrip('/')}/_done/shard-{i:03d}.done" for i in range(args.num_servers)]
    print("[plan] Clearing stale done markers …")
    for u in done_uris:
        b, k = parse_s3_uri(u)
        try:
            s3.delete_object(Bucket=b, Key=k)
        except botocore.exceptions.ClientError:
            pass

    # -------- 2) Mint S3 creds for workers --------
    worker_creds = mint_worker_s3_creds(
        boto_session=session,
        aws_region=args.aws_region,
        input_prefix_s3=args.s3_input_prefix,
        output_prefix_s3=args.s3_output_prefix,
        role_arn=args.assume_role_arn,
        duration_seconds=args.cred_ttl_min * 60,
    )

    # -------- 3) Launch Hetzner servers --------
    hc = HClient(token=args.hcloud_token)

    # Resolve resources
    loc = hc.locations.get_by_name(args.location)
    if loc is None:
        raise RuntimeError(f"Location not found: {args.location}")
    st = hc.server_types.get_by_name(args.server_type)
    if st is None:
        raise RuntimeError(f"Server type not found: {args.server_type}")
    img = hc.images.get_by_name(args.image)
    if img is None:
        # allow passing an image ID
        img = hc.images.get_by_id(int(args.image)) if args.image.isdigit() else None
    if img is None:
        raise RuntimeError(f"Image not found: {args.image}")

    ssh_keys = []
    if args.ssh_key_names:
        for name in [x.strip() for x in args.ssh_key_names.split(",") if x.strip()]:
            k = hc.ssh_keys.get_by_name(name)
            if k is None and name.isdigit():
                k = hc.ssh_keys.get_by_id(int(name))
            if k is None:
                raise RuntimeError(f"SSH key not found: {name}")
            ssh_keys.append(k)

    network = None
    if args.network_id:
        network = hc.networks.get_by_id(args.network_id)
        if network is None:
            raise RuntimeError(f"Network not found: {args.network_id}")

    servers = []
    print(f"[launch] Creating {args.num_servers} server(s) in {args.location} type={args.server_type} image={args.image} …")
    for i in range(args.num_servers):
        udata = build_user_data_for_hetzner(
            aws_region=args.aws_region,
            shard_index=i,
            manifest_s3_uri=manifest_uris[i],
            input_prefix_s3=args.s3_input_prefix,
            output_prefix_s3=args.s3_output_prefix,
            code_zip_s3_uri=args.s3_code_zip,
            tokenizer_cli_args=args.tokenizer_args,
            aws_creds=worker_creds,
            max_dl_workers=args.dl_workers,
        )
        name = f"tok-{i:03d}-{rand_suffix()}"
        resp = hc.servers.create(
            name=name,
            server_type=st,
            image=img,
            location=loc,
            user_data=udata,
            ssh_keys=ssh_keys or None,
            networks=[network] if network else None,
            labels={"project":"midi-tokenizer","shard":f"{i:03d}"},
            start_after_create=True,
        )
        servers.append(resp.server)
        print(f"  - {name} (id={resp.server.id})")

    # -------- 4) Wait for done markers --------
    timeout_s = args.timeout_min * 60 if args.timeout_min else None
    print("[wait] Waiting for done markers …")
    ok = wait_for_done_markers(s3, done_uris, poll_s=30, timeout_s=timeout_s)
    print("[wait] status:", "OK (all shards done)" if ok else "TIMEOUT")

    # -------- 5) Tear down --------
    print("[cleanup] Deleting servers …")
    for s in servers:
        try:
            hc.servers.delete(s)
        except Exception as e:
            print(f"  - delete failed for {s.id}: {e}", file=sys.stderr)

    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
