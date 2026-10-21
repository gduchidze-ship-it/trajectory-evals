"""Run the jailbreak evaluation command line as python -m trajectory_evals.suites.jailbreak."""

from dotenv import load_dotenv

from trajectory_evals.suites.jailbreak.cli import main

if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main())
