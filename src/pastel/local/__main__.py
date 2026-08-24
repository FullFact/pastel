"""Report which questions the local backend can really answer.

python -m pastel.local
"""

from pastel.local.model_registry import report

if __name__ == "__main__":
    report()
