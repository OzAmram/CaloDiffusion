import pytest 

def pytest_addoption(parser):
    parser.addoption("--data-dir", default='./data/', help='Add a specific data dir to use during tests')
    parser.addoption("--calochallenge", default='.', help='Add a specific dir for the CaloChallenge dir is located')
    parser.addoption("--hgcalshowers", default='.', help='Add a specific dir for the HGCalShowers dir is located')

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hgcal: Test only runs on hgcal data (deselect with -m 'not hgcal')"
    )