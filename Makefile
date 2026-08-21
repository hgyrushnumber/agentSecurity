# agentSecurity one-command shortcuts
#
#   make setup   # install environment (add WITH_SFT=1 for training deps)
#   make up      # start API + worker in background
#   make down    # stop everything
#   make logs    # tail API/worker logs
#   make status  # show process status

setup:
	bash scripts/setup.sh $(if $(WITH_SFT),--with-sft,)

up:
	bash scripts/start.sh

down:
	bash scripts/stop.sh

logs:
	tail -f logs/api.log logs/worker.log

status:
	@echo "--- api ---";  if [ -f logs/api.pid ]; then  ps -p $$(cat logs/api.pid) -o pid,command || echo "not running";  else echo "no pid file"; fi
	@echo "--- worker ---"; if [ -f logs/worker.pid ]; then ps -p $$(cat logs/worker.pid) -o pid,command || echo "not running"; else echo "no pid file"; fi

.PHONY: setup up down logs status
