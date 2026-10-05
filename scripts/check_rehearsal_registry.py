#!/usr/bin/env python3
"""Reject a nonprivate GHCR rehearsal destination before publication."""
import argparse
import json
import os
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--allow-missing', action='store_true')
args = parser.parse_args()
request = urllib.request.Request(
    'https://api.github.com/users/nilspalumbo/packages/container/sasy-test',
    headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
             'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'},
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        package = json.load(response)
except urllib.error.HTTPError as error:
    if error.code == 404 and args.allow_missing:
        print('Package absent or inaccessible; first publication defaults private.')
    else:
        raise SystemExit(f'Cannot verify private package: HTTP {error.code}') from None
else:
    if package.get('visibility') != 'private':
        raise SystemExit('Refusing to publish rehearsal into a nonprivate package')
    print('Verified private rehearsal package')
