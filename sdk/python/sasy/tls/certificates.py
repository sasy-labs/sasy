"""
Certificate management for TLS connections with mkcert and self-signed support.
"""

import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


class MkcertError(Exception):
    """Raised when mkcert is not available or not configured properly."""
    pass


def get_mkcert_ca_path() -> Path:
    """
    Get the mkcert CA root directory.
    
    Raises:
        MkcertError: If mkcert is not installed or not initialized
    """
    if not shutil.which("mkcert"):
        raise MkcertError(
            "mkcert is not installed. Please install mkcert from https://github.com/FiloSottile/mkcert"
        )
    
    try:
        result = subprocess.run(
            ["mkcert", "-CAROOT"],
            capture_output=True,
            text=True,
            check=True
        )
        ca_root = Path(result.stdout.strip())
        ca_cert = ca_root / "rootCA.pem"
        
        if not ca_cert.exists():
            raise MkcertError(
                "mkcert root CA not found. "
                "Please run 'mkcert -install' to create and install the root CA"
            )
        
        return ca_cert
    except subprocess.CalledProcessError as e:
        raise MkcertError(f"Failed to get mkcert CA root: {e}")


def generate_mkcert_certificates(
    cert_name: str = "localhost",
    cert_dir: Path = Path("certs"),
    additional_hosts: list[str] | None = None,
) -> dict[str, Path]:
    """
    Generate certificates using mkcert, or return existing paths if they exist.
    
    Args:
        cert_name: Base name for the certificate files
        cert_dir: Directory where certificates will be stored
        additional_hosts: Additional hostnames/IPs to include in the certificate
    
    Returns:
        Dictionary with paths to ca_cert, server_cert, and server_key
        
    Raises:
        MkcertError: If mkcert is not available or not configured
    """
    # Verify mkcert is available and configured
    ca_cert = get_mkcert_ca_path()
    
    cert_dir.mkdir(parents=True, exist_ok=True)
    
    cert_path = cert_dir / f"{cert_name}.crt"
    key_path = cert_dir / f"{cert_name}.key"
    
    # Return existing certificates if they exist
    if cert_path.exists() and key_path.exists():
        print(f"Using existing mkcert certificates for {cert_name}")
        return {
            "ca_cert": ca_cert,
            "server_cert": cert_path,
            "server_key": key_path,
        }
    
    # Build list of hosts
    hosts = [cert_name]
    if cert_name == "localhost":
        hosts.extend(["127.0.0.1", "::1"])
    if additional_hosts:
        hosts.extend(additional_hosts)
    
    try:
        print(f"Generating mkcert certificate for: {', '.join(hosts)}")
        subprocess.run(
            [
                "mkcert",
                "-cert-file", str(cert_path),
                "-key-file", str(key_path),
            ] + hosts,
            check=True,
            capture_output=True,
            text=True
        )
        
        return {
            "ca_cert": ca_cert,
            "server_cert": cert_path,
            "server_key": key_path,
        }
    except subprocess.CalledProcessError as e:
        raise MkcertError(f"Failed to generate certificate with mkcert: {e.stderr}")


def generate_self_signed_cert(
    cert_path: Path,
    key_path: Path,
    ca_cert_path: Path | None = None,
    ca_key_path: Path | None = None,
    common_name: str = "localhost",
    san_dns_names: list[str] | None = None,
    san_ip_addresses: list[str] | None = None,
    days_valid: int = 365,
    organization: str = "SASY development",
) -> tuple[Path, Path]:
    """
    Generate a self-signed certificate or a certificate signed by a CA.
    
    Args:
        cert_path: Path where the certificate will be saved
        key_path: Path where the private key will be saved
        ca_cert_path: Path to CA certificate (if None, creates self-signed)
        ca_key_path: Path to CA private key (if None, creates self-signed)
        common_name: Common name for the certificate
        san_dns_names: List of DNS names for Subject Alternative Name
        san_ip_addresses: List of IP addresses for Subject Alternative Name
        days_valid: Number of days the certificate is valid
        organization: Organization name in the subject. The default says what
            these certificates are for; they are development credentials this
            machine signs for itself, and nothing reads the name.
    
    Returns:
        Tuple of (certificate_path, key_path)
    """
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Build subject. A locally generated development certificate belongs to
    # whoever generated it, so it carries no country, state or locality.
    subject = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])
    
    # If CA cert/key provided, use them; otherwise self-sign
    if ca_cert_path and ca_key_path:
        with open(ca_cert_path, "rb") as f:
            ca_cert = x509.load_pem_x509_certificate(f.read())
        with open(ca_key_path, "rb") as f:
            ca_key = serialization.load_pem_private_key(f.read(), password=None)
        issuer = ca_cert.subject
        signing_key = ca_key
    else:
        issuer = subject
        signing_key = private_key
    
    # Build certificate
    builder = x509.CertificateBuilder()
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(issuer)
    builder = builder.public_key(private_key.public_key())
    builder = builder.serial_number(x509.random_serial_number())
    builder = builder.not_valid_before(datetime.now(UTC))
    builder = builder.not_valid_after(
        datetime.now(UTC) + timedelta(days=days_valid)
    )
    
    # Add Subject Alternative Name extension
    san_list: list[x509.GeneralName] = []
    if san_dns_names:
        san_list.extend(x509.DNSName(name) for name in san_dns_names)
    if san_ip_addresses:
        san_list.extend(x509.IPAddress(ip_address(ip)) for ip in san_ip_addresses)
    
    if not san_list and common_name == "localhost":
        # Default SANs for localhost
        # Note: IPv6 ::1 excluded due to curl/SSL compatibility issues
        san_list = [
            x509.DNSName("localhost"),
            x509.IPAddress(ip_address("127.0.0.1")),
        ]
    elif not san_list:
        # Use common name as SAN
        san_list = [x509.DNSName(common_name)]
    
    builder = builder.add_extension(
        x509.SubjectAlternativeName(san_list),
        critical=False,
    )
    
    # Add basic constraints for CA cert
    if ca_cert_path is None:
        builder = builder.add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
    
    # Sign certificate
    certificate = builder.sign(signing_key, hashes.SHA256())  # type: ignore[arg-type]
    
    # Save certificate
    print(cert_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cert_path, "wb") as f:
        f.write(certificate.public_bytes(serialization.Encoding.PEM))
    
    # Save private key
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    
    return cert_path, key_path


def generate_ca_and_server_certs(
    base_dir: Path = Path("certs"),
    cert_name: str = "localhost",
    san_dns_names: list[str] | None = None,
    san_ip_addresses: list[str] | None = None,
) -> dict[str, Path]:
    """
    Generate a CA certificate and a server certificate signed by that CA.
    
    Args:
        base_dir: Directory where certificates will be stored
        cert_name: Name for the server certificate files
        san_dns_names: DNS names for server certificate SAN
        san_ip_addresses: IP addresses for server certificate SAN
    
    Returns:
        Dictionary with paths to ca_cert, ca_key, server_cert, and server_key
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    ca_cert_path = base_path / "ca.crt"
    ca_key_path = base_path / "ca.key"
    server_cert_path = base_path / f"{cert_name}.crt"
    server_key_path = base_path / f"{cert_name}.key"
    
    # Generate CA certificate if it doesn't exist
    if not ca_cert_path.exists() or not ca_key_path.exists():
        print("Generating CA certificate...")
        generate_self_signed_cert(
            cert_path=ca_cert_path,
            key_path=ca_key_path,
            common_name="SASY CA",
            days_valid=3650,  # 10 years for CA
        )
    
    # Generate server certificate
    print(f"Generating server certificate for {cert_name}...")
    generate_self_signed_cert(
        cert_path=server_cert_path,
        key_path=server_key_path,
        ca_cert_path=ca_cert_path,
        ca_key_path=ca_key_path,
        common_name=cert_name,
        san_dns_names=san_dns_names,
        san_ip_addresses=san_ip_addresses,
        days_valid=365,
    )
    
    return {
        "ca_cert": ca_cert_path,
        "ca_key": ca_key_path,
        "server_cert": server_cert_path,
        "server_key": server_key_path,
    }


def load_credential_strings(cert_path: Path, key_path: Path) -> tuple[bytes, bytes]:
    """
    Load certificate and key as byte strings for gRPC.
    
    Args:
        cert_path: Path to certificate file
        key_path: Path to private key file
    
    Returns:
        Tuple of (certificate_bytes, key_bytes)
    """
    with open(cert_path, "rb") as f:
        cert = f.read()
    with open(key_path, "rb") as f:
        key = f.read()
    return cert, key


def load_tls_credentials(
    cert_path: str | None = None,
    key_path: str | None = None,
    ca_path: str | None = None,
    use_mkcert: bool = False,
    cert_name: str = "localhost",
    cert_dir: str = "certs",
) -> tuple[bytes, bytes, bytes]:
    """
    Load TLS credentials from various sources.
    
    Priority:
    1. Use provided certificate paths if given
    2. Use mkcert if requested and available
    3. Generate self-signed certificates
    
    Args:
        cert_path: Direct path to server certificate
        key_path: Direct path to server private key
        ca_path: Direct path to CA certificate
        use_mkcert: Whether to use mkcert for certificate generation
        cert_name: Name for generated certificates (default: "localhost")
        cert_dir: Directory for generated certificates
    
    Returns:
        Tuple of (server_cert_bytes, server_key_bytes, ca_cert_bytes)
        
    Raises:
        MkcertError: If use_mkcert=True but mkcert is not properly configured
        FileNotFoundError: If provided paths don't exist
    """
    # Option 1: Use provided paths
    if cert_path and key_path and ca_path:
        cert_p = Path(cert_path)
        key_p = Path(key_path)
        ca_p = Path(ca_path)
        
        if not cert_p.exists():
            raise FileNotFoundError(f"Certificate not found: {cert_path}")
        if not key_p.exists():
            raise FileNotFoundError(f"Private key not found: {key_path}")
        if not ca_p.exists():
            raise FileNotFoundError(f"CA certificate not found: {ca_path}")
        
        with open(cert_p, "rb") as f:
            cert = f.read()
        with open(key_p, "rb") as f:
            key = f.read()
        with open(ca_p, "rb") as f:
            ca = f.read()
        
        return cert, key, ca
    
    # Option 2: Use mkcert
    if use_mkcert:
        certs = generate_mkcert_certificates(
            cert_name=cert_name,
            cert_dir=Path(cert_dir)
        )
        
        with open(certs["server_cert"], "rb") as f:
            cert = f.read()
        with open(certs["server_key"], "rb") as f:
            key = f.read()
        with open(certs["ca_cert"], "rb") as f:
            ca = f.read()
        
        return cert, key, ca
    
    # Option 3: Generate self-signed
    print("Generating self-signed certificates...")
    certs = generate_ca_and_server_certs(
        base_dir=Path(cert_dir),
        cert_name=cert_name
    )
    
    with open(certs["server_cert"], "rb") as f:
        cert = f.read()
    with open(certs["server_key"], "rb") as f:
        key = f.read()
    with open(certs["ca_cert"], "rb") as f:
        ca = f.read()
    
    return cert, key, ca


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--mkcert":
        try:
            certs = generate_mkcert_certificates()
            print("\nGenerated mkcert certificates:")
            for name, path in certs.items():
                print(f"  {name}: {path}")
        except MkcertError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        certs = generate_ca_and_server_certs()
        print(f"\nGenerated self-signed certificates in {Path('certs').absolute()}:")
        for name, path in certs.items():
            print(f"  {name}: {path}")
        print("\nTo use these certificates:")
        print("  1. Install ca.crt as a trusted root certificate on clients")
        print("  2. Use server certificates for the gRPC server")
