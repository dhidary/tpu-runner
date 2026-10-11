#!/usr/bin/python3 -I
"""Root-owned, read-only attempt check installed by worker startup."""

import os
from pathlib import Path
import re
import socket
import sys


def main():
    if os.geteuid() != 0:
        raise PermissionError("root-visible process inspection required")
    if len(sys.argv) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", sys.argv[1]):
        raise ValueError("expected one attempt ID")
    # sudo supplies this identity; callers cannot choose another user's UID.
    owner_uid = int(os.environ["SUDO_UID"])
    if owner_uid < 0:
        raise ValueError("invalid sudo caller")
    tag = ("ATTEMPT_ID=" + sys.argv[1]).encode()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.stat().st_uid != owner_uid:
                continue
            if tag in (proc / "environ").read_bytes().split(b"\0"):
                print("TPU_RUNNER_ATTEMPT_BUSY", socket.gethostname())
                return
        except (FileNotFoundError, ProcessLookupError):
            continue
    print("TPU_RUNNER_ATTEMPT_STOPPED", socket.gethostname())


if __name__ == "__main__":
    main()
