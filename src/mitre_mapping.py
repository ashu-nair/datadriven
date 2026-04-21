# src/mitre_mapping.py

ATTACK_TO_CATEGORY = {
    # Normal
    "normal": "Normal",

    # DoS attacks → Impact
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "apache2": "DoS",
    "worm": "DoS",

    # Probe attacks → Discovery
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    # R2L → Initial Access
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",

    # U2R → Privilege Escalation
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}

# Map attack category to MITRE ATT&CK tactic
CATEGORY_TO_MITRE = {
    "Normal": {
        "tactic": "No Threat",
        "tactic_id": "—",
        "technique": "Benign Traffic",
        "technique_id": "—",
        "color": "#2ECC71",
        "description": "No malicious activity detected.",
    },
    "DoS": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Endpoint Denial of Service",
        "technique_id": "T1499",
        "color": "#E74C3C",
        "description": "Adversary is trying to manipulate, interrupt, or destroy systems and data.",
    },
    "Probe": {
        "tactic": "Discovery",
        "tactic_id": "TA0007",
        "technique": "Network Service Scanning",
        "technique_id": "T1046",
        "color": "#F39C12",
        "description": "Adversary is trying to figure out your environment by scanning the network.",
    },
    "R2L": {
        "tactic": "Initial Access",
        "tactic_id": "TA0001",
        "technique": "Exploit Public-Facing Application",
        "technique_id": "T1190",
        "color": "#9B59B6",
        "description": "Adversary is trying to get into your network from a remote location.",
    },
    "U2R": {
        "tactic": "Privilege Escalation",
        "tactic_id": "TA0004",
        "technique": "Exploitation for Privilege Escalation",
        "technique_id": "T1068",
        "color": "#C0392B",
        "description": "Adversary is trying to gain higher-level permissions on your system.",
    },
}

CATEGORIES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
