import subprocess
from pathlib import Path

captures = Path("captures")
out = Path("zeek_out")

for pcap in captures.glob("*.pcap"):
    out_dir = out / pcap.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "docker", "run", "--rm",
        "-v", f"{captures.absolute()}:/pcap",
        "-v", f"{out.absolute()}:/out",
        "zeek/zeek:latest",
        "zeek",
        "-C",
        "-r", f"/pcap/{pcap.name}",
        f"Log::default_logdir=/out/{pcap.stem}"
    ])