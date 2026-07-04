# Gunicorn reads this file automatically when started from the repo root
# (e.g. "gunicorn app:app"), so these settings apply even if Render's Start
# Command passes no CLI flags. This is the robust home for them; render.yaml's
# startCommand is only used when the service is driven by the Blueprint.

# One worker keeps memory within the 512MB Render plan.
#
# Upgrade path when the app feels slow under concurrent uploads: with 1 worker
# only one upload is analyzed at a time (others queue ~10-25s) — that queueing,
# not a crash, is the first wall you'll hit. Raise this to 2 ONLY after moving
# to a larger instance with more RAM: each worker loads its own librosa+numba
# (~250-350MB), so two workers do NOT fit in 512MB and will OOM. So the trigger
# to spend money is concurrency, and the thing to buy is more RAM (next Render
# tier up), which then lets workers = 2. If traffic is bursty rather than
# sustained, moving analysis to a background job is the cheaper alternative to
# upgrading. Don't pre-emptively bump this without the RAM headroom.
workers = 1

# The numba JIT compile runs once at worker boot (app.py -> processor.warmup())
# and is slow on Render's shared CPU. This timeout must be long enough that the
# worker is never killed mid-compile — a mid-compile kill corrupts numba's cache
# and makes later requests fail with "no compiled object yet". After warmup,
# windowed requests finish in a few seconds, well within this budget.
timeout = 300
