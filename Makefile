tests:
	- py.test-3 --cov m3d --cov-report term --verbose
html:
	- py.test-3 --cov m3d --cov-report html --cov-report term --verbose
