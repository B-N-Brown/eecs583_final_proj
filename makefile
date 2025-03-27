-include secrets.mk
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip3

install:
	$(PIP) install --no-cache-dir -r requirements.txt

pip_list:
	$(PIP) list

send_to_great_lakes:
	rsync -avz --exclude '.venv' --exclude '.gitignore' --exclude '.git' --exclude '__pycache__' --exclude 'secrets.mk' ./ $(GREAT_LAKES):$(REMOTE_FILE)