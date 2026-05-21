---
name: drivevla-windows-mount
description: "Windows host mounts remote /root/DriveVLA-W0 from 163.61.150.40:60024 to Z: via sshfs-win. Helper scripts at C:\\Users\\zhido\\bin\\drivevla\\."
metadata: 
  node_type: memory
  type: project
  originSessionId: 98b4f070-7b4f-4a6d-8973-a813a1142b76
---

Remote project `/root/DriveVLA-W0` on `163.61.150.40:60024` (user `root`, host `xt-p-A100-013`, 8x A100) is mounted to **`Z:\`** on this Windows machine via sshfs-win.

**Why:** User wants to edit remote code locally with Claude Code so changes sync to the server in realtime — same pattern as their Linux-side `remote.md` (which uses sshfs + fusermount3).

**How to apply:**
- Use `Z:\` paths in Claude Code (Read/Edit/Write) — writes propagate to remote immediately.
- DO NOT run training, large IO, or python interpreters on `Z:\` — every read goes over SSH. Heavy work happens on the remote via `ssh new-server "..."`. SSH alias `new-server` is configured in `~/.ssh/config`.
- For batch sync (push code, pull logs), prefer `rsync` over `ssh new-server` rather than copying via `Z:\`.

**Key gotchas discovered during setup (don't repeat):**
- The bundled sshfs in `C:\Program Files\SSHFS-Win\bin\sshfs.exe` calls `ssh` by name. If Windows OpenSSH `ssh.exe` is found first on PATH, sftp subsystem gets "Connection reset by peer". Fix: pass `-o ssh_command=C:/PROGRA~1/SSHFS-Win/bin/ssh.exe` (8.3 short name — sshfs option parser splits on spaces, so quoting "Program Files" doesn't work).
- For `IdentityFile`, use forward-slash Windows path `C:/Users/zhido/.ssh/id_ed25519`. Backslashes get stripped by PowerShell argument passing; `/cygdrive/c/...` isn't mounted in the minimal bundled cygwin.
- `\\sshfs\` UNC mount (`net use Z: \\sshfs\root@host!port\path`) defaults to password auth and prompts — bypass by invoking `sshfs.exe` directly with explicit key.

**Helper scripts** (PowerShell, no admin needed once installed):
- `C:\Users\zhido\bin\drivevla\mount.ps1` — start sshfs, mount Z:
- `C:\Users\zhido\bin\drivevla\unmount.ps1` — kill sshfs, unmount

**State files:** `%LOCALAPPDATA%\drivevla-mount\sshfs.pid` + sshfs.err.log/sshfs.out.log

**Auth:** ed25519 keypair at `~/.ssh/id_ed25519` (`SHA256:8ZsneHxVLnMuWVCJKQ2bX9TlDRra5s3wK6o8fLRJ+0I`), public key appended to remote `/root/.ssh/authorized_keys`.

**Mount is NOT persistent across reboot.** Re-run `mount.ps1` after restarting Windows, or wire it into Task Scheduler at logon if needed.
