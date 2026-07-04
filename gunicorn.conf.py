# Gunicorn reads this file automatically when started from the repo root
# (e.g. "gunicorn app:app"), so these settings apply even if Render's Start
# Command passes no CLI flags. This is the robust home for them; render.yaml's
# startCommand is only used when the service is driven by the Blueprint.

# One worker keeps memory within the 512MB Render plan.
workers = 1

# The numba JIT compile runs once at worker boot (app.py -> processor.warmup())
# and is slow on Render's shared CPU. This timeout must be long enough that the
# worker is never killed mid-compile — a mid-compile kill corrupts numba's cache
# and makes later requests fail with "no compiled object yet". After warmup,
# windowed requests finish in a few seconds, well within this budget.
timeout = 300
