"""Run the jailbreak eval command line: python -m trajectory_evals.suites.jailbreak."""

from dotenv import load_dotenv

from trajectory_evals.suites.jailbreak.cli import main

if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main())
