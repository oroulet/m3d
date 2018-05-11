all:
	- py.test-3 --cov m3d --cov-report html --cov-report term --cov-report xml
