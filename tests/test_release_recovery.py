"""Recovery must preserve original release qualification and immutable provenance."""
import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery', ROOT / 'scripts/prepare_release_recovery.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SHA = 'a' * 40


def source():
    run = {'repository': {'id': 1405994907, 'private': False}, 'event': 'push',
           'status': 'completed', 'path': '.github/workflows/sdk-python-release.yml',
           'head_branch': 'sasy-v0.5.0', 'head_sha': SHA}
    jobs = [{'name': name, 'conclusion': 'success'} for name in module.REQUIRED_JOBS]
    artifacts = [{'name': name, 'expired': False} for name in module.REQUIRED_ARTIFACTS]
    return run, jobs, artifacts


def test_qualified_release_can_recover():
    assert module.validate_source(*source(), SHA) == '0.5.0'


@pytest.mark.parametrize('problem', ['private', 'other_repo', 'branch', 'pr', 'running',
                                    'other_workflow', 'moved_tag', 'failed_gate',
                                    'missing_gate', 'expired', 'missing_artifact', 'duplicate_artifact'])
def test_unqualified_or_unavailable_sources_fail_closed(problem):
    run, jobs, artifacts = copy.deepcopy(source())
    tag_sha = SHA
    if problem == 'private': run['repository']['private'] = True
    elif problem == 'other_repo': run['repository']['id'] = 1
    elif problem == 'branch': run['head_branch'] = 'main'
    elif problem == 'pr': run['event'] = 'pull_request'
    elif problem == 'running': run['status'] = 'in_progress'
    elif problem == 'other_workflow': run['path'] = '.github/workflows/ci.yml'
    elif problem == 'moved_tag': tag_sha = 'b' * 40
    elif problem == 'failed_gate': jobs[0]['conclusion'] = 'failure'
    elif problem == 'missing_gate': jobs.pop()
    elif problem == 'expired': artifacts[0]['expired'] = True
    elif problem == 'missing_artifact': artifacts.pop()
    elif problem == 'duplicate_artifact': artifacts.append(artifacts[0])
    with pytest.raises(ValueError): module.validate_source(run, jobs, artifacts, tag_sha)


def test_receipt_requires_one_unambiguous_digest():
    digest = 'sha256:' + 'a' * 64
    assert module.publication_digest('publish\t2026-10-05T00:00:00Z ' + digest + '\n') == digest
    for log in ['no push receipt', digest + '\nsha256:' + 'b' * 64]:
        with pytest.raises(ValueError): module.publication_digest(log)


def test_recovery_skips_builds_and_uses_original_artifacts_and_publisher():
    workflow = yaml.load((ROOT / '.github/workflows/sdk-python-release.yml').read_text(), Loader=yaml.BaseLoader)
    jobs = workflow['jobs']
    assert "inputs.recovery_run == ''" in jobs['core']['if']
    assert "inputs.recovery_run == ''" in jobs['build']['if']
    assert jobs['publish_recovered']['needs'] == 'recover_image'
    assert jobs['publish_recovered']['environment']['name'] == 'pypi'
    for job_name in ['recover_image', 'publish_recovered']:
        steps = jobs[job_name]['steps']
        assert all('uv build' not in step.get('run', '') and 'docker build' not in step.get('run', '') for step in steps)
        downloads = [step for step in steps if step.get('uses', '').startswith('actions/download-artifact@')]
        assert downloads and all(step['with']['run-id'] == '${{ inputs.recovery_run }}' for step in downloads)
