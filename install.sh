#!/bin/sh
# AllForOne installer for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/ahmedmsmg/AllForOne/main/install.sh | sh -s -- <join-code>
#
# The join code is a Tailscale auth key (tskey-auth-...) from whoever hosts the group. Without
# one, this machine keeps whatever Tailscale network it is already on.
#
# What it does:
#   1. installs the AllForOne programs that match this OS and GPU into ~/.allforone/bin
#   2. installs Tailscale if needed and joins the group's network with the join code
#   3. starts the AllForOne agent in the background, now and at every login
set -eu

REPO="ahmedmsmg/AllForOne"
KEY="${1:-}"
DIR="$HOME/.allforone"
BIN="$DIR/bin"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

have curl || die "curl is required"
have tar || die "tar is required"

# NVIDIA builds are made per GPU generation (compute capability) with CUDA 12.8, so they need a
# matching card and a driver that supports CUDA 12.8 or newer. Prints the build name, or fails.
cuda_variant() {
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -n1)
  major=${cuda% *}; minor=${cuda#* }
  if [ -z "$cuda" ] || [ "$major" -lt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -lt 8 ]; }; then
    say "NVIDIA driver supports CUDA ${major:-?}.${minor:-?}; the GPU build needs 12.8+. Using the CPU build (update the driver to use the GPU)." >&2
    return 1
  fi
  case "$cap" in
    75 | 80 | 86 | 89 | 90 | 120) echo "$1-x86_64-cuda-sm$cap" ;;
    *) say "No GPU build for compute capability ${cap:-?}; using the CPU build." >&2; return 1 ;;
  esac
}

# --- which build ---------------------------------------------------------------------------
os=$(uname -s)
arch=$(uname -m)
case "$os/$arch" in
  Linux/x86_64)
    variant=linux-x86_64-cpu
    if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
      variant=$(cuda_variant linux) || variant=linux-x86_64-cpu
    fi
    ;;
  Linux/aarch64 | Linux/arm64) variant=linux-aarch64-cpu ;;
  Darwin/arm64) variant=macos-arm64-metal ;;
  Darwin/x86_64) variant=macos-x86_64-cpu ;;
  *) die "unsupported system: $os/$arch" ;;
esac

fetch_build() {
  url="https://github.com/$REPO/releases/latest/download/afo-$1.tar.gz"
  say "Downloading AllForOne ($1)"
  tmp=$(mktemp -d)
  if ! curl -fL --progress-bar -o "$tmp/afo.tar.gz" "$url"; then
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$BIN"
  mkdir -p "$BIN"
  tar -xzf "$tmp/afo.tar.gz" -C "$BIN"
  rm -rf "$tmp"
}
if ! fetch_build "$variant"; then
  case "$variant" in
    *-cuda-*) say "No GPU build available; using the CPU build"; variant=linux-x86_64-cpu; fetch_build "$variant" ;;
    *) die "could not download the AllForOne build for $variant" ;;
  esac
fi
[ "$os" = Darwin ] && xattr -dr com.apple.quarantine "$BIN" 2>/dev/null || true

# Put the programs on PATH for new shells.
line="export PATH=\"$BIN:\$PATH\""
for rc in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc"; do
  if [ -f "$rc" ] || [ "$(basename "$rc")" = .profile ]; then
    grep -qsF "$BIN" "$rc" || printf '\n# AllForOne\n%s\n' "$line" >> "$rc"
  fi
done
export PATH="$BIN:$PATH"

# --- Tailscale ------------------------------------------------------------------------------
if [ "$os" = Darwin ]; then
  app=/Applications/Tailscale.app/Contents/MacOS/Tailscale
  if [ ! -x "$app" ]; then
    if have brew; then
      say "Installing Tailscale"
      brew install --cask tailscale
    else
      die "install Tailscale from https://tailscale.com/download/mac, open it once, then run this installer again"
    fi
  fi
  open -a Tailscale || true
  # The app's binary doubles as the command-line tool.
  ln -sf "$app" "$BIN/tailscale"
  ts() { "$BIN/tailscale" "$@"; }
else
  if ! have tailscale; then
    say "Installing Tailscale (asks for your password)"
    curl -fsSL https://tailscale.com/install.sh | sh
  fi
  ts() { sudo tailscale "$@"; }
  # Tailscale's daemon must be running; the official installer enables it under systemd.
  if have systemctl; then sudo systemctl enable --now tailscaled >/dev/null 2>&1 || true; fi
fi

if [ -n "$KEY" ]; then
  say "Joining the group's network"
  ts up --auth-key="$KEY" --hostname="$(hostname | tr -cs 'A-Za-z0-9-' '-' | sed 's/-*$//')" \
    || die "tailscale up failed; check the join code"
elif ! tailscale status >/dev/null 2>&1; then
  say "Tailscale isn't connected yet. Run this installer with the group's join code, or run: tailscale up"
fi

# Let tailnet traffic through a host firewall that would otherwise drop it.
if [ "$os" = Linux ] && have ufw && sudo ufw status 2>/dev/null | grep -q "Status: active"; then
  sudo ufw allow in on tailscale0 >/dev/null
fi

# --- agent in the background ----------------------------------------------------------------
mkdir -p "$DIR/logs"
if [ "$os" = Darwin ]; then
  plist="$HOME/Library/LaunchAgents/com.allforone.agent.plist"
  mkdir -p "$(dirname "$plist")"
  cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.allforone.agent</string>
  <key>ProgramArguments</key><array><string>$BIN/afo</string><string>agent</string></array>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$BIN:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$DIR/logs/agent.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/agent.log</string>
</dict>
</plist>
EOF
  launchctl bootout "gui/$(id -u)/com.allforone.agent" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$plist"
elif have systemctl && systemctl --user show-environment >/dev/null 2>&1; then
  unit="$HOME/.config/systemd/user/allforone-agent.service"
  mkdir -p "$(dirname "$unit")"
  cat > "$unit" <<EOF
[Unit]
Description=AllForOne agent
After=network-online.target

[Service]
ExecStart=$BIN/afo agent
Environment=PATH=$BIN:/usr/local/bin:/usr/bin:/bin
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable allforone-agent >/dev/null 2>&1
  systemctl --user restart allforone-agent
  # Keep the agent running when this user isn't logged in, where allowed.
  loginctl enable-linger "$(id -un)" >/dev/null 2>&1 || true
else
  pkill -x afo 2>/dev/null || true
  nohup "$BIN/afo" agent > "$DIR/logs/agent.log" 2>&1 &
  say "No service manager found; the agent runs until you log out. Start it again with: afo agent"
fi

say "Done. This machine is now offering itself to the group ($variant)."
echo "    To start a session from this machine:  afo host"
echo "    Agent log:                              $DIR/logs/agent.log (or: journalctl --user -u allforone-agent)"
