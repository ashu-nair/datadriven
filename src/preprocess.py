# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.mitre_mapping import ATTACK_TO_CATEGORY

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

FEATURE_COLS = [c for c in NSL_KDD_COLUMNS if c not in ("label", "difficulty_level")]

# Fixed encoding maps so train/test/app stay consistent
PROTOCOL_TYPES = ["tcp", "udp", "icmp"]
SERVICES = [
    "http", "ftp_data", "smtp", "ssh", "ftp", "pop_3", "telnet",
    "domain_u", "finger", "sunrpc", "auth", "eco_i", "ntp_u",
    "ecr_i", "other", "private", "domain", "mtp", "IRC",
    "X11", "Z39_50", "aol", "bgp", "courier", "csnet_ns",
    "ctf", "daytime", "discard", "echo", "efs", "exec",
    "gopher", "harvest", "hostnames", "http_2784", "http_443",
    "http_8001", "imap4", "iso_tsap", "klogin", "kshell", "ldap",
    "link", "login", "netbios_dgm", "netbios_ns", "netbios_ssn",
    "netstat", "nnsp", "nntp", "pm_dump", "pop_2", "printer",
    "red_i", "remote_job", "rje", "shell", "sql_net", "supdup",
    "systat", "tim_i", "time", "urh_i", "urp_i", "uucp",
    "uucp_path", "vmnet", "whois",
]
FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTR", "SH", "OTH",
         "S1", "S2", "S3", "RSTOS0"]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # protocol_type
    df["protocol_type"] = pd.Categorical(
        df["protocol_type"], categories=PROTOCOL_TYPES
    ).codes
    # service
    df["service"] = pd.Categorical(
        df["service"], categories=SERVICES
    ).codes
    # flag
    df["flag"] = pd.Categorical(
        df["flag"], categories=FLAGS
    ).codes
    return df


def load_nslkdd(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None, names=NSL_KDD_COLUMNS)
    return df


def map_label(label: str) -> str:
    label = label.strip().lower()
    return ATTACK_TO_CATEGORY.get(label, "Normal")


def prepare_data(df: pd.DataFrame):
    """Return X (features), y (encoded labels), label_names list."""
    df = df.copy()

    # Map raw attack labels → category
    df["category"] = df["label"].apply(map_label)

    # Encode categoricals with fixed maps
    df = encode_categoricals(df)

    X = df[FEATURE_COLS].fillna(0)
    y_raw = df["category"]

    # Encode categories to integers
    le = LabelEncoder()
    le.fit(["DoS", "Normal", "Probe", "R2L", "U2R"])
    y = le.transform(y_raw)

    return X, y, list(le.classes_)
