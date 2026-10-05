"""Recover only the retained artifacts of a qualified public version-tag release."""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

REPOSITORY = 'sasy-labs/sasy'
REQUIRED_JOBS = {
    'build', 'engine / version', 'engine / core / core',
    'engine / build (ubuntu-24.04, amd64, x86_64)',
    'engine / build (ubuntu-24.04-arm, arm64, aarch64)', 'engine / publish',
}
REQUIRED_ARTIFACTS = {'engine-image-amd64', 'engine-image-arm64', 'sasy-python-package'}


def validate_source(run, jobs, artifacts, tag_sha):
    if (run['repository']['id'] != 1405994907 or run['repository']['private']
            or run['event'] != 'push' or run['status'] != 'completed'
            or run['path'] != '.github/workflows/sdk-python-release.yml'
            or not re.fullmatch(r'sasy-v\d+\.\d+\.\d+', run['head_branch'])
            or run['head_sha'] != tag_sha):
        raise ValueError('Recovery requires a completed public version-tag release at its original commit')
    for name in REQUIRED_JOBS:
        matching = [job for job in jobs if job['name'] == name]
        if len(matching) != 1 or matching[0]['conclusion'] != 'success':
            raise ValueError(f'Required release gate did not succeed: {name}')
    for name in REQUIRED_ARTIFACTS:
        matching = [artifact for artifact in artifacts if artifact['name'] == name]
        if len(matching) != 1 or matching[0]['expired']:
            raise ValueError(f'Required release artifact is unavailable: {name}')
    return run['head_branch'].removeprefix('sasy-v')


def publication_digest(log):
    receipts = re.findall(r'(?:^|\s)(sha256:[0-9a-f]{64})\s*$', log, re.MULTILINE)
    if len(set(receipts)) != 1:
        raise ValueError('Expected one immutable image digest in the successful publication receipt')
    return receipts[0]


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id', type=int)
    args = parser.parse_args()
    root = f'repos/{REPOSITORY}'
    run = api(f'{root}/actions/runs/{args.run_id}')
    tag = run['head_branch']
    if not re.fullmatch(r'sasy-v\d+\.\d+\.\d+', tag):
        raise ValueError('Not a version-tag release')
    obj = api(f'{root}/git/ref/tags/{tag}')['object']
    while obj['type'] == 'tag':
        obj = api(f"{root}/git/tags/{obj['sha']}")['object']
    if obj['type'] != 'commit':
        raise ValueError('Release tag must resolve to a commit')
    jobs = api(f"{root}/actions/runs/{args.run_id}/attempts/{run['run_attempt']}/jobs?per_page=100")['jobs']
    artifacts = api(f'{root}/actions/runs/{args.run_id}/artifacts?per_page=100')['artifacts']
    version = validate_source(run, jobs, artifacts, obj['sha'])
    job = next(job for job in jobs if job['name'] == 'engine / publish')
    log = subprocess.check_output(['gh', 'run', 'view', '--repo', REPOSITORY,
                                   '--job', str(job['id']), '--log'], text=True)
    digest = publication_digest(log)
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        output.write(f"version={version}\nsha={run['head_sha']}\ntag={tag}\ndigest={digest}\n")
    print(f'Recovering {tag} from run {args.run_id}, published index {digest}')


if __name__ == '__main__':
    main()
