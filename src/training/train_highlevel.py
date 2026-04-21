"""High-level training entrypoint (PPO selector over frozen low-level policies).

Beginner note:
1) Freeze all low-level SAC policies.
2) PPO action selects one of 4 low-level policies.
3) Add switch penalty to avoid thrashing.
"""

from __future__ import annotations


def main() -> None:
    print("[highlevel] TODO: wire PPO selector after low-level SAC checkpoints are ready.")


if __name__ == "__main__":
    main()
