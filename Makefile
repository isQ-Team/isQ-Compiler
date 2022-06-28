frontend:
	cd frontend && stack build
simulator:
	cd simulator && cargo build --release
.PHONY: frontend simulator