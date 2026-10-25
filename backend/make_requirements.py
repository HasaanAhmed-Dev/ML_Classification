"""
Writes backend/requirements.txt pinned to the versions installed in YOUR environment.
A saved scikit-learn model only loads reliably with the same scikit-learn version that trained it,
so we pin it. Run from the project root:   python backend/make_requirements.py
"""
from importlib.metadata import version
from pathlib import Path

PACKAGES = ["flask", "pandas", "numpy", "scikit-learn", "scipy", "joblib", "requests", "pytest"]
lines = [f"{name}=={version(name)}" for name in PACKAGES]
out = Path(__file__).resolve().parent / "requirements.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out}:")
print("\n".join(lines))
