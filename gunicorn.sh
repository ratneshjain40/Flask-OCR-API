#!/bin/bash

gunicorn server:app -b 0.0.0.0:8000