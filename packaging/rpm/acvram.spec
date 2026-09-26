Name:           acvram
Version:        @VERSION@
Release:        1%{?dist}
Summary:        OpenAI-API-compatible inference gateway (per-GPU quantisation, VRAM->VRAM->RAM tiering)
License:        GPL-3.0-or-later
URL:            https://github.com/anticitoyun/anticitoyen-vram
Source0:        %{url}/archive/v%{version}/anticitoyen-vram-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3
Requires:       python3 >= 3.10, python3-pip

# Fedora choice, not Arch's: Fedora carries no python3-torch-cuda (or equivalent) in its
# official repos — RPM Fusion and Fedora's own python3-torch build are CPU-only. Depending
# on a system torch package here would either be silently CPU-only (wrong for a GPU gateway)
# or unresolvable. Instead this package ships the SAME bootstrap launcher as the .deb
# (tools/construire-deb.sh): first run creates a private venv under
# $ACVRAM_HOME (default ~/.local/share/acvram) and installs the torch build matching the
# GPUs actually present (CPU / cu124 / cu130), exactly as `./install.sh` does from source.
# Open question sent to duck.ai (poste4, 26/09): is there a better Fedora-native path
# (a COPR torch-cuda build, RPM Fusion nonfree) worth switching to later.
Requires:       python3-numpy, python3-pillow

%description
An OpenAI-API-compatible inference gateway that treats memory as a hierarchy, gives each
GPU the numeric format its silicon reads best (NVFP4, INT4, INT8), and optimises every
token in joules as much as in seconds. See the README for the two-GPU rig it targets and
the numbers behind each default.

%prep
%autosetup -n anticitoyen-vram-%{version}

%build

%install
install -d %{buildroot}%{_datadir}/acvram %{buildroot}%{_bindir}
cp -r acvram pyproject.toml README.md LICENSE %{buildroot}%{_datadir}/acvram/
install -Dm644 LICENSE %{buildroot}%{_datadir}/licenses/%{name}/LICENSE
install -Dm755 packaging/rpm/acvram-lanceur %{buildroot}%{_bindir}/acvram

%files
%license LICENSE
%{_bindir}/acvram
%{_datadir}/acvram/
%{_datadir}/licenses/%{name}/

%post
echo "acvram installed. First run bootstraps a private venv (torch chosen for the GPUs present, several GB) — 'acvram doctor' checks it."

%changelog
* Fri Sep 26 2026 AntiCitoyen <anticitoyen@users.noreply.gitlab.com> - @VERSION@-1
- See https://github.com/anticitoyun/anticitoyen-vram/releases
