#!/bin/sh
python download_models.py
exec uvicorn main:app --host 0.0.0.0 --port 8000
