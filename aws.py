#!/usr/bin/env python3
import argparse, base64, boto3, botocore, gzip, io, json, os, sys, time, textwrap, random
from urllib.parse import urlparse
from collections import defaultdict

# --------------------
# Helpers
# --------------------
def parse_s3_uri(uri: str):
    u = urlparse(uri)
    if u.scheme != "s3" or not u.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = u.netloc
    key = u.path.lstrip("/")
    return bucket, key

def s3_put_jsonl_gz(s3, bucket, key, rows_iter):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for row in rows_iter:
            line = (row if isinstance(row, str) else json.dumps(row)).rstrip("\n") + "\n"
            gz.write(line.encode("utf-8"))
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="application/json", ContentEncoding="gzip")

def s3_list_objects(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # skip "directory markers"
            if obj["Key"].endswith("/"):
                continue
            yield obj["Key"], obj["Size"]

def greedy_balance(objects, n_bins):
    """
    objects: list of (key, size)
    returns: list[b] -> list[(key,size)] approximately balanced by total size
    """
    bins = [dict(size=0, items=[]) for _ in range(n_bins)]
    # biggest first
    for k, sz in sorted(objects, key=lambda x: x[1], reverse=True):
        bins.sort(key=lambda b: b["size"])  # smallest bin first
        bins[0]["items"].append((k, sz))
        bins[0]["size"] += sz
    return [b["items"] for b in bins]

def wait_for_done_markers(s3, done_uris, poll_s=30, timeout_s=None):
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
                if code not in ("404", "NoSuchKey", "NotFound"):
                    raise
        if not remaining:
            return True
        if timeout_s and (time.time() - start) > timeout_s:
            return False
        time.sleep(poll_s)
    return True

# --------------------
# User-data (cloud-init) for each instance
# --------------------
def build_user_data(
    region,
    shard_index,
    manifest_s3_uri,
    input_prefix_s3,
    output_prefix_s3,
    code_zip_s3_uri,
    tokenizer_cli_args,
    venv_dir="/opt/worker/venv",
    work_dir="/opt/worker",
    max_dl_workers=32,
):
    """
    Returns a valid cloud-init user-data string that:
      - installs deps
      - pulls code + requirements
      - downloads this shard's files (from manifest)
      - runs tokenizer.py
      - syncs outputs and writes a _done marker
    """

    shard_name = f"shard-{shard_index:03d}"

    # 1) Build the inner bash script first (single f-string).
    script_body = textwrap.dedent(f"""
        set -euo pipefail

        REGION="{region}"
        MANIFEST_URI="{manifest_s3_uri}"
        INPUT_S3_PREFIX="{input_prefix_s3.rstrip('/')}"
        OUTPUT_S3_PREFIX="{output_prefix_s3.rstrip('/')}"
        CODE_ZIP_URI="{code_zip_s3_uri}"
        SHARD="{shard_name}"
        WORK_DIR="{work_dir}"
        VENV_DIR="{venv_dir}"
        DATA_DIR="/mnt/data/{shard_name}"
        OUT_DIR="/mnt/out/{shard_name}"
        LOG_DIR="/var/log/tokenizer"
        DL_WORKERS="{int(max_dl_workers)}"

        mkdir -p "$WORK_DIR" "$DATA_DIR" "$OUT_DIR" "$LOG_DIR"
        aws configure set default.region "$REGION" || true

        # Install packages
        if command -v dnf >/dev/null 2>&1; then PKG="dnf";
        elif command -v yum >/dev/null 2>&1; then PKG="yum";
        else PKG="apt-get"; fi
        sudo $PKG -y update || true
        sudo $PKG -y install python3 python3-venv unzip jq awscli git || true

        # Python env
        python3 -m venv "$VENV_DIR"
        . "$VENV_DIR/bin/activate"
        pip install --upgrade pip wheel setuptools

        # Pull code
        aws s3 cp "$CODE_ZIP_URI" "$WORK_DIR/code.zip"
        unzip -q -o "$WORK_DIR/code.zip" -d "$WORK_DIR/code"

        # Install requirements (fallback minimal set)
        if [ -f "$WORK_DIR/code/requirements.txt" ]; then
          pip install -r "$WORK_DIR/code/requirements.txt"
        else
          pip install numpy mido pretty_midi boto3
        fi

        # Use multiprocessing; keep BLAS single-threaded
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        ulimit -n 1048576 || true

        # Download manifest
        aws s3 cp "$MANIFEST_URI" "$WORK_DIR/manifest.jsonl.gz"

        # Downloader: concurrent small-file pulls
        cat > "$WORK_DIR/download_by_manifest.py" << 'PY'
import json, gzip, os, sys, concurrent.futures, boto3
from urllib.parse import urlparse
import botocore

manifest_gz = sys.argv[1]
dest_root = sys.argv[2]
max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32

def parse_s3(uri):
    u = urlparse(uri)
    if u.scheme != "s3": raise ValueError("bad uri: " + uri)
    return u.netloc, u.path.lstrip("/")

s3 = boto3.client("s3")

def dl(one):
    u = one.strip()
    if not u: return
    b,k = parse_s3(u)
    rel = k.split("/", 1)[1] if "/" in k else os.path.basename(k)
    dest = os.path.join(dest_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        s3.download_file(b, k, dest)
        return dest
    except botocore.exceptions.ClientError as e:
        print(f"[WARN] download failed {{u}}: {{e}}", file=sys.stderr)
        return None

def rows():
    with gzip.open(manifest_gz, "rt") as f:
        for line in f:
            j = json.loads(line)
            yield j["s3_uri"]

def main():
    todo = list(rows())
    print(f"Downloading {{len(todo)}} files to {{dest_root}}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, _ in enumerate(ex.map(dl, todo, chunksize=32), 1):
            if i % 500 == 0:
                print(f"  ... {{i}} / {{len(todo)}}")
    print("Done downloads.")
if __name__ == "__main__":
    main()
PY

        python "$WORK_DIR/download_by_manifest.py" "$WORK_DIR/manifest.jsonl.gz" "$DATA_DIR" "$DL_WORKERS"

        # Run tokenizer
        cd "$WORK_DIR/code"

        CORES=$(nproc || echo 16)
        W=$((CORES>8 ? CORES-4 : CORES))

        echo "Running tokenizer with $W workers"
        set +e
        python tokenizer.py \
          --midi-root "$DATA_DIR" \
          --out-dir "$OUT_DIR" \
          --num-workers "$W" \
          --log-level INFO \
          {tokenizer_cli_args} | tee "$LOG_DIR/$SHARD.log"
        RC=$?
        set -e

        # Upload outputs & mark done (even if RC!=0 we ship what's there)
        aws s3 sync "$OUT_DIR" "$OUTPUT_S3_PREFIX/$SHARD/" --no-progress

        python - <<PY
import json, os
d = {{"shard": "{shard_name}", "exit_code": {{}}, "host": os.uname().nodename}}
open("{work_dir}/DONE.json","w").write(json.dumps(d))
PY

        # Replace the {{}} placeholder with RC (portable)
        sed -i "s/{{}}/$RC/g" "{work_dir}/DONE.json"

        aws s3 cp "{work_dir}/DONE.json" "$OUTPUT_S3_PREFIX/_done/{shard_name}.done"
        echo "DONE with code $RC"
    """).strip()

    # 2) Escape single quotes for the YAML inline exec array:  - [ bash, -lc, '<script>' ]
    #    Use the classic bash-safe split: 'foo'"'"'bar'
    script_escaped = script_body.replace("'", "'\"'\"'")

    # 3) Assemble cloud-init (no nested f-strings).
    user_data = (
        "#cloud-config\n"
        "runcmd:\n"
        f"  - [ bash, -lc, '{script_escaped}' ]\n"
    )

    return user_data

# --------------------
# Main Orchestrator
# --------------------
def main():
    ap = argparse.ArgumentParser(description="Launch c7i.48xlarge fleet and tokenize MIDI shards")
    ap.add_argument("--region", required=True)
    ap.add_argument("--num-instances", type=int, required=True)
    ap.add_argument("--instance-type", default="c7i.48xlarge")
    ap.add_argument("--ami-id", required=False, help="AMZ Linux 2023 or Ubuntu AMI ID; if omitted, tries AL2023 lookup")
    ap.add_argument("--subnet-id", required=True)
    ap.add_argument("--security-group-id", required=True)
    ap.add_argument("--iam-instance-profile", required=True, help="Name or ARN with S3 read/write permissions")
    ap.add_argument("--key-name", required=False, help="Optional: SSH key pair name")
    ap.add_argument("--ebs-volume-size", type=int, default=200, help="GiB for root volume")
    ap.add_argument("--use-spot", action="store_true", help="Use spot instances")
    ap.add_argument("--spot-max-price", default=None, help="Optional max price for spot")
    ap.add_argument("--s3-input-prefix", required=True, help="s3://bucket/prefix where MIDIs live")
    ap.add_argument("--s3-output-prefix", required=True, help="s3://bucket/prefix to upload tokenized shards")
    ap.add_argument("--s3-code-zip", required=True, help="s3://bucket/path/code_artifact.zip with tokenizer.py & requirements.txt")
    ap.add_argument("--tokenizer-args", default="", help="Extra CLI args passed to tokenizer.py")
    ap.add_argument("--manifest-prefix", required=False, help="s3://bucket/prefix to store manifests; default <output>/_manifests/")
    ap.add_argument("--timeout-min", type=int, default=0, help="Orchestration timeout (0=none)")
    args = ap.parse_args()

    region = args.region
    ec2 = boto3.client("ec2", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    # 0) Lookup AMI if not provided (Amazon Linux 2023)
    ami = args.ami_id
    if not ami:
        images = ec2.describe_images(
            Owners=["amazon"],
            Filters=[{"Name":"name","Values":["al2023-ami-*-x86_64"]},{"Name":"state","Values":["available"]}]
        )["Images"]
        if not images:
            print("No AL2023 AMI found; please provide --ami-id", file=sys.stderr); sys.exit(2)
        ami = sorted(images, key=lambda x: x["CreationDate"], reverse=True)[0]["ImageId"]

    # 1) List input objects
    in_bucket, in_prefix = parse_s3_uri(args.s3_input_prefix)
    print(f"Listing objects under s3://{in_bucket}/{in_prefix} ...", flush=True)
    # Only take *.tar.gz / *.tgz objects as the work units
    objs = [
        (k, sz)
        for (k, sz) in s3_list_objects(s3, in_bucket, in_prefix)
        if k.lower().endswith((".tar.gz", ".tgz"))
    ]
    if not objs:
        print("No objects found under input prefix.", file=sys.stderr); sys.exit(2)
    total_bytes = sum(sz for _, sz in objs)
    print(f"Found {len(objs)} files, total {total_bytes/1e9:.2f} GB")

    # 2) Partition into shards
    n = args.num_instances
    shards = greedy_balance(objs, n)
    out_bucket, out_prefix = parse_s3_uri(args.s3_output_prefix)
    manifest_bucket, manifest_prefix = parse_s3_uri(args.manifest_prefix or f"s3://{out_bucket}/{out_prefix.rstrip('/')}/_manifests/")
    # ensure trailing slash
    if not manifest_prefix.endswith("/"): manifest_prefix += "/"

    manifest_uris = []
    for i, shard in enumerate(shards):
        def rows():
            for key, size in shard:
                yield {"s3_uri": f"s3://{in_bucket}/{key}", "size": int(size)}
        man_key = f"{manifest_prefix}shard-{i:03d}.jsonl.gz"
        print(f"Uploading manifest {i+1}/{n} -> s3://{manifest_bucket}/{man_key}")
        s3_put_jsonl_gz(s3, manifest_bucket, man_key, rows())
        manifest_uris.append(f"s3://{manifest_bucket}/{man_key}")

    # 3) Launch instances
    code_zip_uri = args.s3_code_zip
    print(f"Launching {n} x {args.instance_type} in {region} (AMI {ami})")
    instances_to_shard = {}
    user_datas = []
    for i in range(n):
        ud = build_user_data(
            region=region,
            shard_index=i,
            manifest_s3_uri=manifest_uris[i],
            input_prefix_s3=args.s3_input_prefix,
            output_prefix_s3=args.s3_output_prefix,
            code_zip_s3_uri=code_zip_uri,
            tokenizer_cli_args=args.tokenizer_args,
        )
        user_datas.append(ud)

    # Prepare common params
    inst_params = dict(
        ImageId=ami,
        InstanceType=args.instance_type,
        KeyName=args.key_name if args.key_name else None,
        NetworkInterfaces=[{
            "DeviceIndex": 0,
            "SubnetId": args.subnet_id,
            "Groups": [args.security_group_id],
            "AssociatePublicIpAddress": True,
        }],
        IamInstanceProfile={"Name": args.iam_instance_profile} if not args.iam_instance_profile.startswith("arn:") else {"Arn": args.iam_instance_profile},
        MinCount=1, MaxCount=1,
        BlockDeviceMappings=[{
            "DeviceName": "/dev/xvda",
            "Ebs": {"VolumeSize": args.ebs_volume_size, "VolumeType": "gp3", "DeleteOnTermination": True}
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key":"Project","Value":"midi-tokenizer"},
                     {"Key":"Role","Value":"tokenizer-worker"}]
        }],
    )

    instance_ids = []
    for i, ud in enumerate(user_datas):
        params = dict(inst_params)
        params["UserData"] = ud
        if args.use_spot:
            params["InstanceMarketOptions"] = {
                "MarketType":"spot",
                "SpotOptions": {"MaxPrice": args.spot_max_price} if args.spot_max_price else {}
            }
        resp = ec2.run_instances(**{k:v for k,v in params.items() if v is not None})
        inst = resp["Instances"][0]
        iid = inst["InstanceId"]
        instance_ids.append(iid)
        instances_to_shard[iid] = i
        # tag each with shard index for clarity
        ec2.create_tags(Resources=[iid], Tags=[{"Key":"Shard","Value":f"{i:03d}"}])
        print(f"  Launched {iid} -> shard-{i:03d}")

    # 4) Wait for completion (watch S3 _done markers), then terminate
    done_uris = [f"{args.s3_output_prefix.rstrip('/')}/_done/shard-{i:03d}.done" for i in range(n)]
    print("Waiting for done markers in S3:")
    for u in done_uris: print("  ", u)
    timeout_s = args.timeout_min*60 if args.timeout_min else None
    ok = wait_for_done_markers(s3, done_uris, poll_s=60, timeout_s=timeout_s)
    if not ok:
        print("WARNING: Timeout waiting for completion. You may have partial results.", file=sys.stderr)

    print("Terminating instances...")
    # terminate in batches (max 100 per API call)
    for i in range(0, len(instance_ids), 100):
        ec2.terminate_instances(InstanceIds=instance_ids[i:i+100])
    print("All terminate requests sent.")

    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
