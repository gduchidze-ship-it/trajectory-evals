"""Run the jailbreak evaluation command line as python -m evals.jailbreak."""

from dotenv import load_dotenv

from evals.jailbreak.cli import main

if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main())
