"""Public tests use local fixtures; engine/toolchain checks are explicit lanes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true",
                     help="Run owned-engine tests; requires SASY_TEST_ENGINE")
    parser.addoption("--run-toolchain", action="store_true",
                     help="Run policy toolchain tests; missing tools are failures")


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    for item in items:
        excluded = any(item.get_closest_marker(marker) is not None
                       and not config.getoption(option)
                       for marker, option in (("integration", "--run-integration"),
                                              ("toolchain", "--run-toolchain")))
        (deselected if excluded else selected).append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
