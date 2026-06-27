"""
Master Pipeline - Intelligent IDS for Encrypted Network Traffic
Runs all phases (1-9) sequentially and reports pass/fail for each.
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import subprocess
import sys
import time

SCRIPTS = [
    ("Phase 1 — Data Preprocessing",        "scripts/preprocess.py"),
    ("Phase 2 — Exploratory Data Analysis",  "scripts/eda.py"),
    ("Phase 3 — Baseline ML Models",         "scripts/train_baseline.py"),
    ("Phase 4 — Feature Optimisation",       "scripts/feature_optimization.py"),
    ("Phase 5 — Autoencoder Training",       "scripts/train_autoencoder.py"),
    ("Phase 6 — Hybrid IDS Engine",          "scripts/hybrid_ids.py"),
    ("Phase 7 — Model Evaluation",           "scripts/evaluate.py"),
    ("Phase 8 — Generalization Test",        "scripts/generalization_test.py"),
    ("Phase 9 — SHAP Explainability",        "scripts/shap_explain.py"),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(msg):
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

def run_phase(label, script):
    banner(label)
    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # stream output live
        text=True
    )
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n{GREEN}[OK]  {label} completed in {elapsed:.1f}s{RESET}")
        return True
    else:
        print(f"\n{RED}[FAIL]  {label} FAILED (exit code {result.returncode}){RESET}")
        return False

def main():
    banner("INTELLIGENT IDS — FULL PIPELINE START")
    print(f"Working directory: {os.getcwd()}\n")

    results = []
    total_start = time.time()

    for label, script in SCRIPTS:
        ok = run_phase(label, script)
        results.append((label, ok))

        if not ok:
            print(f"\n{YELLOW}[WARN]  Pipeline halted at: {label}{RESET}")
            print("Fix the error above and re-run the pipeline.\n")
            break

    total_elapsed = time.time() - total_start

    banner("PIPELINE SUMMARY")
    all_ok = True
    for label, ok in results:
        icon  = f"{GREEN}[OK]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
        print(f"  {icon}  {label}")
        if not ok:
            all_ok = False

    print(f"\n  Total time: {total_elapsed/60:.1f} minutes")

    if all_ok:
        print(f"\n{BOLD}{GREEN}[DONE]  All phases completed successfully!{RESET}")
        print(f"{GREEN}        Check the results/ folder for all plots and metrics.{RESET}\n")
    else:
        print(f"\n{RED}Pipeline did not complete fully - see errors above.{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
