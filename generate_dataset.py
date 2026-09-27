import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_PER_CLASS = 700
rows = []

protocols = ["tcp", "udp", "icmp"]
services = ["http", "https", "dns", "ssh", "ftp", "smtp", "other"]

def normal_row():
    duration = rng.gamma(2.0, 1.8)
    src_bytes = max(40, rng.lognormal(7.1, 0.7))
    dst_bytes = max(30, rng.lognormal(7.0, 0.75))
    packets = max(2, rng.poisson(18))
    src_port = rng.integers(1024, 65535)
    dst_port = int(rng.choice([53, 80, 443, 22, 25, 21, 8080]))
    failed = rng.binomial(2, 0.04)
    conn_rate = max(0.1, rng.gamma(2, 2))
    unique_ports = max(1, rng.poisson(1.3))
    syn = rng.poisson(2)
    rst = rng.poisson(0.3)
    http_req = rng.poisson(3) if dst_port in [80, 443, 8080] else 0
    protocol = rng.choice(protocols, p=[0.72, 0.23, 0.05])
    service = rng.choice(services)
    return [duration, src_bytes, dst_bytes, packets, protocol, service, src_port,
            dst_port, failed, conn_rate, unique_ports,
            src_bytes / packets, syn, rst, http_req, "BENIGN"]

def dos_row():
    duration = rng.uniform(0.01, 2)
    packets = rng.poisson(180) + 30
    src_bytes = rng.lognormal(6.2, 0.5)
    dst_bytes = rng.lognormal(5.0, 0.5)
    return [duration, src_bytes, dst_bytes, packets, "tcp",
            rng.choice(["http", "https", "other"]), rng.integers(1024, 65535),
            int(rng.choice([80, 443, 8080])), rng.poisson(1),
            rng.uniform(40, 220), rng.integers(1, 5),
            src_bytes / packets, rng.poisson(60), rng.poisson(8),
            rng.poisson(20), "DoS"]

def portscan_row():
    packets = rng.poisson(35) + 5
    src_bytes = rng.lognormal(6, 0.5)
    return [rng.uniform(0.01, 1.5), src_bytes, rng.lognormal(5.5, .5),
            packets, "tcp", "other", rng.integers(1024, 65535),
            rng.integers(1, 65535), rng.poisson(1), rng.uniform(15, 90),
            rng.integers(20, 300), src_bytes / packets, rng.poisson(18),
            rng.poisson(5), 0, "PortScan"]

def brute_row():
    packets = rng.poisson(25) + 5
    src_bytes = rng.lognormal(6.5, .4)
    return [rng.uniform(.1, 5), src_bytes, rng.lognormal(5.8, .5),
            packets, "tcp", "ssh", rng.integers(1024, 65535), 22,
            rng.integers(4, 15), rng.uniform(5, 35), rng.integers(1, 4),
            src_bytes / packets, rng.poisson(5), rng.poisson(2),
            0, "BruteForce"]

def webattack_row():
    packets = rng.poisson(45) + 8
    src_bytes = rng.lognormal(7.2, .6)
    return [rng.uniform(.2, 8), src_bytes, rng.lognormal(6.8, .6),
            packets, "tcp", rng.choice(["http", "https"]), rng.integers(1024, 65535),
            int(rng.choice([80, 443, 8080])), rng.poisson(2),
            rng.uniform(4, 25), rng.integers(1, 8), src_bytes / packets,
            rng.poisson(8), rng.poisson(2), rng.poisson(25), "WebAttack"]

def botnet_row():
    packets = rng.poisson(12) + 3
    src_bytes = rng.lognormal(6.3, .7)
    return [rng.uniform(.1, 12), src_bytes, rng.lognormal(6.1, .6),
            packets, rng.choice(["tcp", "udp"]), rng.choice(["dns", "http", "other"]),
            rng.integers(1024, 65535), int(rng.choice([53, 80, 443, 8080])),
            rng.poisson(2), rng.uniform(2, 18), rng.integers(3, 15),
            src_bytes / packets, rng.poisson(4), rng.poisson(2),
            rng.poisson(6), "Botnet"]

generators = [normal_row, dos_row, portscan_row, brute_row, webattack_row, botnet_row]
for gen in generators:
    for _ in range(N_PER_CLASS):
        rows.append(gen())

columns = [
    "duration", "src_bytes", "dst_bytes", "packets", "protocol", "service",
    "src_port", "dst_port", "failed_connections", "connection_rate",
    "unique_dst_ports", "bytes_per_packet", "syn_count", "rst_count",
    "http_requests", "label"
]

df = pd.DataFrame(rows, columns=columns)
for col in ["duration", "src_bytes", "dst_bytes", "packets", "src_port", "dst_port",
            "failed_connections", "connection_rate", "unique_dst_ports",
            "bytes_per_packet", "syn_count", "rst_count", "http_requests"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
out = Path(__file__).parent / "data" / "network_traffic.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)

print(f"Generated {len(df):,} rows -> {out}")
print(df["label"].value_counts())

def main():
    return df

if __name__ == "__main__":
    main()
