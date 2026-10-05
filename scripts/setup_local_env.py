"""Fill missing local SDK settings without requiring an installed SDK."""
import argparse
import json
import os
import re
from pathlib import Path

# Consume entire quoted values, including multiline ones, so their contents
# cannot be mistaken for settings. Keep existing text verbatim.
BINDING = re.compile(
    r"^[ \t]*(?:export[ \t]+)?(?P<key>'[^'\n]+'|[^\s=#]+)[ \t]*"
    r"(?:=[ \t]*(?:'(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"|[^\r\n]*))?"
    r"[^\r\n]*(?:\r?\n|$)", re.MULTILINE,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    path = Path('.env')
    if path.is_symlink():
        parser.error('Refusing to write through a .env symlink')
    existing = path.read_text() if path.exists() else ''
    present = {match['key'].strip("'").upper() for match in BINDING.finditer(existing)}
    settings = {
        'SASY_URL': args.url,
        'TLS_CA_PATH': str(Path('certs/server.crt').resolve()),
    }
    if 'SASY_API_KEY' not in present:
        config = json.loads(Path('config/auth/apikey.json').read_text())
        clients = [key for key, role in config['static_keys'].items() if role == 'client']
        if len(clients) != 1:
            parser.error('Set SASY_API_KEY in .env: expected exactly one client key in config/auth/apikey.json')
        settings['SASY_API_KEY'] = clients[0]
    missing = {key: value for key, value in settings.items() if key not in present}
    # Single-quoted dotenv values preserve backslashes, spaces and # characters.
    # Dollar interpolation must not change a literal path or an existing key.
    if any('\n' in value or '\r' in value or '${' in value for value in missing.values()):
        parser.error('Local settings cannot contain line breaks or ${...} interpolation')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        if missing:
            if existing and not existing.endswith('\n'):
                stream.write('\n')
            for key, value in missing.items():
                escaped = value.replace('\\', '\\\\').replace("'", "\\'")
                stream.write(f"{key}='{escaped}'\n")
    print('Local SDK settings ready in .env (existing values preserved)')


if __name__ == '__main__':
    main()
