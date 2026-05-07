import time
import random
import shlex
import uuid
import docker
import json
import os

from datetime import datetime
from docker.errors import NotFound

client = docker.from_env()

NETWORK_NAME = "stealth-dos-simulator_dosnet"

all_running = []


ATTACK_IP_POOL = [
    f"172.30.0.{i}"
    for i in range(10, 250)
]


def get_used_ips():

    network = client.networks.get(NETWORK_NAME)

    containers = network.attrs.get("Containers", {})

    used = set()

    for c in containers.values():
        ip = c.get("IPv4Address")
        if ip:
            used.add(ip.split("/")[0])

    return used


def allocate_ip():

    used_ips = get_used_ips()

    available = [
        ip for ip in ATTACK_IP_POOL
        if ip not in used_ips
    ]

    if not available:
        raise RuntimeError("No available IPs in subnet")

    return random.choice(available)


def save_result(event, path="container_results.json"):

    def safe(o):
        if isinstance(o, datetime):
            return o.isoformat()
        return o

    clean_event = {
        "name": event["name"],
        "attack_type": event["attack_type"],
        "intensity": event["intensity"],
        "ip": event["ip"],
        "connections": event["connections"],
        "start_ts": event["start_ts"],
        "start_dt": safe(event["start_dt"]),
        "end_ts": event.get("end_ts"),
        "end_dt": safe(event.get("end_dt")) if event.get("end_dt") else None,
        "duration": event.get("duration"),
    }

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(clean_event)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def build_command(mode, connections, duration):

    return (
        f"-g -o results "
        f"{mode} "
        f"-c {connections} "
        f"-x 5 -i 5 -r 3 "
        f"-l {duration} "
        f"-u http://target_nginx:80/rudy"
    )


def create_networking_config(ip):

    return client.api.create_networking_config({
        NETWORK_NAME: client.api.create_endpoint_config(
            ipv4_address=ip
        )
    })

def launch_attack(attack_type, mode, intensity):

    duration = random.randint(120, 320)

    if intensity == "high":
        connections = random.randint(100, 200)
    else:
        connections = random.randint(20, 60)

    ip = allocate_ip()

    command = build_command(mode, connections, duration)

    args = shlex.split(command)

    name = f"{attack_type}_{uuid.uuid4().hex[:8]}"

    start_dt = datetime.now()
    start_ts = time.time()

    networking_config = create_networking_config(ip)

    container_info = client.api.create_container(
        image="shekyan/slowhttptest:latest",
        command=args,
        name=name,
        networking_config=networking_config,
        host_config=client.api.create_host_config(),
        detach=True,
        tty=False
    )

    container_id = container_info["Id"]

    client.api.start(container_id)

    print(
        f"[START] {name} | {attack_type} | "
        f"{intensity} | {ip}"
    )

    return {
        "name": name,
        "container_id": container_id,
        "attack_type": attack_type,
        "intensity": intensity,
        "connections": connections,
        "ip": ip,
        "start_ts": start_ts,
        "start_dt": start_dt,
        "end_ts": None,
        "end_dt": None,
        "duration": None
    }

def finalize(events):

    for event in events:

        try:

            container = client.containers.get(event["container_id"])

            container.wait()

            container.reload()

            finished_at = container.attrs["State"]["FinishedAt"]

            end_dt = datetime.fromisoformat(
                finished_at.replace("Z", "+00:00")
            )

            end_ts = end_dt.timestamp()

            event["end_ts"] = end_ts
            event["end_dt"] = end_dt
            event["duration"] = end_ts - event["start_ts"]

            save_result(event)

            print(f"[END] {event['name']} | {event['ip']}")

            container.remove(force=True)

        except NotFound:
            print(f"[MISSING] {event['name']}")

        except Exception as e:
            print(f"[ERROR] {event['name']} -> {e}")


attack_profiles = [
    ("rudy", "-B"),
    ("slowloris", "-H"),
    ("slowread", "-X")
]

for _ in range(12):

    attack_type, mode = random.choice(attack_profiles)

    intensity = random.choice(["low", "high"])

    event = launch_attack(
        attack_type,
        mode,
        intensity
    )

    all_running.append(event)

    time.sleep(random.randint(30, 80))


finalize(all_running)

print(f"Collected events: {len(all_running)}")